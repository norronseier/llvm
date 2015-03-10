#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "glicm"

// Copied from LICM.cpp
static bool inSubLoop(BasicBlock *BB, Loop *CurLoop, LoopInfo *LI);

namespace {
  struct GLICM : public LoopPass {
    static char ID;
    GLICM() : LoopPass(ID) {
      initializeGLICMPass(*PassRegistry::getPassRegistry());
    }

    bool runOnLoop(Loop *L, LPPassManager &LPM) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequired<ScalarEvolution>();
      AU.addRequiredID(LoopSimplifyID);
    }

  private:
    int tmpCount = 0;

    DominatorTree *DT;
    LoopInfo *LI;
    AliasAnalysis *AA;
    AliasSetTracker *CurAST;
    DenseMap<Loop*, AliasSetTracker*> LoopToAliasSetMap;

    Loop *CurLoop;
    Loop *ParentLoop;
    PHINode *IndVar;
    PHINode *phiNode;
    BasicBlock *NewLoopBody;
    BasicBlock *NewLoopCond;
    BasicBlock *NewLoopPreheader;

    void createMirrorLoop(Loop *L, unsigned TripCount);
    bool generalizedHoist(DomTreeNode *N, LICMSafetyInfo *SafetyInfo);
    bool hasLoopInvariantOperands(Instruction *I, Loop *OuterLoop);
    bool isInvariantForGLICM(Value *V, Loop *OuterLoop);
    void hoist(Instruction *I);
    void replaceScalarsWithArrays(unsigned TripCount);
    bool isUsedInOriginalLoop(Instruction *I);
  };
}

char GLICM::ID = 0;
INITIALIZE_PASS_BEGIN(GLICM, "glicm", "GLICM", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(GLICM, "glicm", "GLICM", false, false)

bool GLICM::runOnLoop(Loop *L, LPPassManager &LPM) {
  ScalarEvolution *SCEV = &getAnalysis<ScalarEvolution>();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  AA = &getAnalysis<AliasAnalysis>();

  dbgs() << "===== Current loop: " << *L << " ========\n";
  CurLoop = L;
  unsigned TripCount = SCEV->getSmallConstantTripCount(L);
  IndVar = L->getCanonicalInductionVariable();
  ParentLoop = L->getParentLoop();

  // Compute loop safety information.
  LICMSafetyInfo SafetyInfo;
  computeLICMSafetyInfo(&SafetyInfo, CurLoop);

  // ============ [Start] Copied from LICM ================
  CurAST = new AliasSetTracker(*AA);
  // Collect Alias info from subloops.
  for (Loop::iterator LoopItr = L->begin(), LoopItrE = L->end();
       LoopItr != LoopItrE; ++LoopItr) {
    Loop *InnerL = *LoopItr;
    AliasSetTracker *InnerAST = LoopToAliasSetMap[InnerL];
    assert(InnerAST && "Where is my AST?");

    // What if InnerLoop was modified by other passes ?
    CurAST->add(*InnerAST);

    // Once we've incorporated the inner loop's AST into ours, we don't need the
    // subloop's anymore.
    delete InnerAST;
    LoopToAliasSetMap.erase(InnerL);
  }

  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    if (LI->getLoopFor(BB) == L)        // Ignore blocks in subloops.
      CurAST->add(*BB);                 // Incorporate the specified basic block
  }

  if (L->getParentLoop())
    LoopToAliasSetMap[L] = CurAST;
  else
    delete CurAST;
  // ============= [End] Copied from LICM =================

  // Apply this to loops that have a canonical indvar (starting from 0 and incrementing by 1
  // on each loop step) and a statically computable trip count.
  if (ParentLoop && ParentLoop->isLoopSimplifyForm() &&
      IndVar && TripCount > 0) {
    if (tmpCount++ > 0)
      return false;
    createMirrorLoop(ParentLoop, TripCount);
    generalizedHoist(DT->getNode(L->getHeader()), &SafetyInfo);
    replaceScalarsWithArrays(TripCount);
  }
  return false;
}

void GLICM::createMirrorLoop(Loop *L, unsigned TripCount) {
  BasicBlock *Header = L->getHeader();
  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Header || !Preheader)
    return;

  // Create two new BBs. One for the loop body, and one where the loop condition
  // is tested.
  NewLoopBody = SplitEdge(Preheader, Header, DT, LI);
  NewLoopCond = SplitEdge(NewLoopBody, Header, DT, LI);
  NewLoopBody->setName(CurLoop->getHeader()->getName() + Twine(".gcm.1"));
  NewLoopCond->setName(CurLoop->getHeader()->getName() + Twine(".gcm.2"));
  NewLoopPreheader = NewLoopBody->getUniquePredecessor();

  // Add a new PhiNode at the end of NewLoopBody. This node corresponds to
  // the induction variable of the new loop
  phiNode = PHINode::Create(IndVar->getType(), 2,
                                     IndVar->getName() + Twine(".gcm"),
                                     NewLoopBody->getTerminator());

  // Increment the induction variable.
  BinaryOperator *nextIndVar = BinaryOperator::
                    CreateAdd(phiNode,
                              ConstantInt::getSigned(phiNode->getType(), 1),
                              phiNode->getName() + Twine(".inc"),
                              NewLoopCond->getTerminator());

  // Insert a conditional branch instruction based on the induction variable.
  CmpInst *CmpInst = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_SLE, nextIndVar,
                  ConstantInt::getSigned(nextIndVar->getType(), TripCount),
                  phiNode->getName() + Twine(".cmp"),
                  NewLoopCond->getTerminator());

  BasicBlock *Dummy = SplitEdge(NewLoopCond, Header, DT, LI);
  Dummy->setName(CurLoop->getHeader()->getName() + Twine(".gcm.end"));

  // Remove the terminator instruction, since it is not needed anymore.
  Dummy->removePredecessor(NewLoopCond);
  NewLoopCond->getTerminator()->eraseFromParent();

  // Create the branch, finalizing the loop.
  BranchInst::Create(NewLoopBody, Dummy, CmpInst, NewLoopCond);

  // Fill in incoming values for the PhiNode.
  phiNode->addIncoming(ConstantInt::getSigned(phiNode->getType(), 0),
                       Preheader);
  phiNode->addIncoming(nextIndVar, NewLoopCond);

  // Construction of the replicated loop is now complete.
  return;
}

bool GLICM::generalizedHoist(DomTreeNode* N, LICMSafetyInfo *SafetyInfo) {
  bool Changed = false;
  BasicBlock *BB = N->getBlock();

  if (!CurLoop->contains(BB))
    return Changed;
  if (CurLoop->getLoopLatch() == BB)
    return Changed;

  if (!inSubLoop(BB, CurLoop, LI)) {
    dbgs() << "In BB:" << BB->getName() << "\n";
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E; ) {
      Instruction &I = *II++;
      if (&I == IndVar)
        // This is the induction variable, do not hoist it
        continue;

      dbgs() << "Instruction: " << I << "\n";
      bool i1 = hasLoopInvariantOperands(&I, ParentLoop);
      bool i2 = canSinkOrHoistInst(I, AA, DT, CurLoop, CurAST, SafetyInfo);
      bool i3 = isSafeToExecuteUnconditionally(I, DT, CurLoop, SafetyInfo);
      dbgs() << i1 << " " << i2 << " " << i3 << "\n";
      /*if (isInvariantInGivenLoop(&I, ParentLoop) &&
          canSinkOrHoistInst(I, AA, DT, CurLoop, CurAST, SafetyInfo) &&
          isSafeToExecuteUnconditionally(I, DT, CurLoop, SafetyInfo)) { */
      if (i1 && i2 && i3) {
        hoist(&I);
        dbgs() << "Would hoist.\n";
      } else
        dbgs() << "Would not hoist.\n";
    }
  }
  const std::vector<DomTreeNode*> &Children = N->getChildren();
  for (unsigned i = 0, e = Children.size(); i != e; ++i)
    Changed |= generalizedHoist(Children[i], SafetyInfo);
  return Changed;
}

bool GLICM::hasLoopInvariantOperands(Instruction *I, Loop *OuterLoop) {
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (!isInvariantForGLICM(I->getOperand(i), OuterLoop))
      return false;

  return true;
}

void GLICM::hoist(Instruction *I) {
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (PHINode *PhiNode = dyn_cast<PHINode>(I->getOperand(i)))
      if (PhiNode == IndVar)
        I->replaceUsesOfWith(IndVar, phiNode);
  I->moveBefore(NewLoopBody->getTerminator());
}

bool GLICM::isInvariantForGLICM(Value *V, Loop *OuterLoop) {
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    if (LI->getLoopFor(I->getParent()) == OuterLoop)
      return false;
    if (PHINode *PhiNode = dyn_cast<PHINode>(I))
      if (PhiNode == IndVar)
        return true;
    return CurLoop->hasLoopInvariantOperands(I);
  }

  return true;
}

void GLICM::replaceScalarsWithArrays(unsigned TripCount) {
  std::vector<Instruction*> instructions;

  // Construct a list of all the instructions that are used in the original
  // loop. These will need to be stored in temporary arrays.
  for (BasicBlock::iterator II = NewLoopBody->begin(), E = NewLoopBody->end();
       II != E; ) {
    Instruction &I = *II++;
    if (&I == phiNode)
      continue;
    if (isUsedInOriginalLoop(&I))
      instructions.push_back(&I);
  }

  // There is nothing to do;
  if (instructions.empty())
    return;

  BasicBlock *TmpArrBlock = SplitEdge(NewLoopPreheader, NewLoopBody, DT, LI);
  TmpArrBlock->setName(NewLoopBody->getName() + Twine(".tmparr"));

  Constant *tmpArrSize = ConstantInt::getSigned(phiNode->getType(), TripCount);
  dbgs() << "Array size: " << *tmpArrSize << "\n";

  for (std::vector<Instruction*>::iterator it = instructions.begin();
       it != instructions.end(); ) {
    Instruction *Instr = *it++;

    // Insert an array allocation before the first non-PHI instruction in the block.
    // Otherwise the constraint that PHI nodes are grouped at the start of a BB
    // is broken.
    AllocaInst *tmpArr = new AllocaInst(Instr->getType(),
                                        tmpArrSize,
                                        Instr->getName() + Twine(".tmparr"),
                                        TmpArrBlock->getTerminator());
                                        //NewLoopBody->getFirstNonPHI());

    SmallVector<Value*, 8> GEPIdx;
    GEPIdx.push_back(phiNode);
    GetElementPtrInst *GEP = GetElementPtrInst::Create(tmpArr, GEPIdx, Instr->getName() + Twine(".idx"),
                              NewLoopBody->getFirstNonPHI());

    StoreInst *store = new StoreInst(Instr, GEP, TmpArrBlock);
    store->removeFromParent();
    store->insertAfter(Instr);

    // Replace use in original loop with array. This needs 1) a GEP operation,
    // 2) a load into a temporary and 3) to replace the uses.
    SmallVector<Value*, 8> GEPIdx2;
    GEPIdx2.push_back(IndVar);
    GetElementPtrInst *GEP2 = GetElementPtrInst::Create(tmpArr, GEPIdx2, Instr->getName() + Twine(".idx.1"),
                              CurLoop->getHeader()->getFirstNonPHI());

    LoadInst *loadInst = new LoadInst(GEP2, Instr->getName() + Twine(".tmparr.elem"),
                                       CurLoop->getHeader()->getFirstNonPHI());
    loadInst->removeFromParent();
    loadInst->insertAfter(GEP2);

    for (User *U : Instr->users()) {
      if (Instruction *I = dyn_cast<Instruction>(U)) {
        if (LI->getLoopFor(I->getParent()) == CurLoop)
          I->replaceUsesOfWith(Instr, loadInst);
      }
    }
    //BasicBlock::iterator ii(Instr);
    //ReplaceInstWithValue(Instr->getParent()->getInstList(), ii, loadInst); */

    // dbgs() << *Instr << " needs to stored in an array.\n";
  }

  instructions.clear();
}

// Returns true if the given instruction (which is part of the cloned loop)
// is used in the current loop.
bool GLICM::isUsedInOriginalLoop(Instruction *I) {
  for (User *U : I->users()) {
    if (Instruction *UserInstr = dyn_cast<Instruction>(U)) {
      if (LI->getLoopFor(UserInstr->getParent()) == CurLoop)
        return true;
    }
  }
  return false;
}

static bool inSubLoop(BasicBlock *BB, Loop *CurLoop, LoopInfo *LI) {
  assert(CurLoop->contains(BB) && "Only valid if BB is IN the loop");
  return LI->getLoopFor(BB) != CurLoop;
}
