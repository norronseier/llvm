#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
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

STATISTIC(NumHoisted,   "Number of instructions hoisted to parent loop's "
                        "pre-header");
STATISTIC(NumTmpArrays, "Number of temporary arrays allocated");

static bool hoist(Instruction *I, BasicBlock *InsertionBlock);
static void replaceIndVarOperand(Instruction *I, PHINode *Old, PHINode *New);
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
    DominatorTree *DT;
    LoopInfo *LI;
    AliasAnalysis *AA;
    AliasSetTracker *CurAST;
    DenseMap<Loop*, AliasSetTracker*> LoopToAliasSetMap;

    Loop *CurLoop;
    Loop *ParentLoop;
    PHINode *IndVar;
    PHINode *NewLoopIndVar;
    BasicBlock *NewLoopHeader;
    BasicBlock *NewLoopLatch;
    BasicBlock *NewLoopPreheader;

    void createMirrorLoop(Loop *L, unsigned TripCount);
    bool generalizedHoist(DomTreeNode *N, LICMSafetyInfo *SafetyInfo);
    bool hasLoopInvariantOperands(Instruction *I, Loop *OuterLoop);
    bool isInvariantForGLICM(Value *V, Loop *OuterLoop);
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

  CurLoop = L;
  unsigned TripCount = SCEV->getSmallConstantTripCount(L);
  IndVar = L->getCanonicalInductionVariable();
  ParentLoop = L->getParentLoop();

  // Compute Loop safety information.
  // FIXME: This function computes LICM safety info for the pre-header of the
  // inspected loop. We want instead to check the pre-header of the parent loop.
  LICMSafetyInfo SafetyInfo;
  computeLICMSafetyInfo(&SafetyInfo, CurLoop);

  // Compute the Alias Set Tracker.
  CurAST = new AliasSetTracker(*AA);
  // Collect Alias information from subloops.
  for (Loop::iterator LoopItr = L->begin(), LoopItrE = L->end();
       LoopItr != LoopItrE; ++LoopItr) {
    Loop *InnerL = *LoopItr;
    AliasSetTracker *InnerAST = LoopToAliasSetMap[InnerL];
    assert(InnerAST && "Where is my AST?");

    CurAST->add(*InnerAST);

    delete InnerAST;
    LoopToAliasSetMap.erase(InnerL);
  }

  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    if (LI->getLoopFor(BB) == L)
      CurAST->add(*BB);
  }

  if (L->getParentLoop())
    LoopToAliasSetMap[L] = CurAST;
  else
    delete CurAST;

  // Apply generalized loop-invariant code motion to loops that satisfy all the
  // following conditions:
  // 1) they have a canonical induction variable (starting from 0 and
  //    incremented by 1 each iteration) and a computable trip count. This is
  //    needed in order to clone the structure of the loop in the parent loop's
  //    pre-header.
  // 2) they have a parent loop which is in loop simplify form (to ensure the
  //    loop has a valid pre-header)
  if (ParentLoop && ParentLoop->isLoopSimplifyForm() && IndVar &&
      TripCount > 0) {
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

  // Create a header and a latch for the new loop.
  NewLoopHeader = SplitEdge(Preheader, Header, DT, LI);
  NewLoopLatch = SplitEdge(NewLoopHeader, Header, DT, LI);
  NewLoopHeader->setName(CurLoop->getHeader()->getName() + Twine(".gcm.1"));
  NewLoopLatch->setName(CurLoop->getHeader()->getName() + Twine(".gcm.2"));
  NewLoopPreheader = NewLoopHeader->getUniquePredecessor();

  // Add a new PhiNode at the end of NewLoopHeader. This node corresponds to
  // the induction variable of the new loop
  NewLoopIndVar = PHINode::Create(IndVar->getType(), 2,
                                     IndVar->getName() + Twine(".gcm"),
                                     NewLoopHeader->getTerminator());

  // Increment the induction variable.
  BinaryOperator *nextIndVar = BinaryOperator::
                    CreateAdd(NewLoopIndVar,
                              ConstantInt::getSigned(NewLoopIndVar->getType(), 1),
                              NewLoopIndVar->getName() + Twine(".inc"),
                              NewLoopLatch->getTerminator());

  // Insert a conditional branch instruction based on the induction variable.
  CmpInst *CmpInst = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_SLE, nextIndVar,
                  ConstantInt::getSigned(nextIndVar->getType(), TripCount),
                  NewLoopIndVar->getName() + Twine(".cmp"),
                  NewLoopLatch->getTerminator());

  BasicBlock *Dummy = SplitEdge(NewLoopLatch, Header, DT, LI);
  Dummy->setName(CurLoop->getHeader()->getName() + Twine(".gcm.end"));

  // Remove the terminator instruction, since it is not needed anymore.
  Dummy->removePredecessor(NewLoopLatch);
  NewLoopLatch->getTerminator()->eraseFromParent();

  // Create the branch, finalizing the loop.
  BranchInst::Create(NewLoopHeader, Dummy, CmpInst, NewLoopLatch);

  // Fill in incoming values for the PhiNode.
  NewLoopIndVar->addIncoming(ConstantInt::getSigned(NewLoopIndVar->getType(), 0),
                       Preheader);
  NewLoopIndVar->addIncoming(nextIndVar, NewLoopLatch);

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

      // dbgs() << "Instruction: " << I << "\n";
      // bool i1 = hasLoopInvariantOperands(&I, ParentLoop);
      // bool i2 = canSinkOrHoistInst(I, AA, DT, CurLoop, CurAST, SafetyInfo);
      // bool i3 = isSafeToExecuteUnconditionally(I, DT, CurLoop, SafetyInfo);
      // dbgs() << i1 << " " << i2 << " " << i3 << "\n";
      if (hasLoopInvariantOperands(&I, ParentLoop) &&
          canSinkOrHoistInst(I, AA, DT, CurLoop, CurAST, SafetyInfo) &&
          isSafeToExecuteUnconditionally(I, DT, CurLoop, SafetyInfo)) {
        hoist(&I, NewLoopHeader);
        replaceIndVarOperand(&I, IndVar, NewLoopIndVar);
      }
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
  for (BasicBlock::iterator II = NewLoopHeader->begin(), E = NewLoopHeader->end();
       II != E; ) {
    Instruction &I = *II++;
    if (&I == NewLoopIndVar)
      continue;
    if (isUsedInOriginalLoop(&I))
      instructions.push_back(&I);
  }

  // There is nothing to do;
  if (instructions.empty())
    return;

  BasicBlock *TmpArrBlock = SplitEdge(NewLoopPreheader, NewLoopHeader, DT, LI);
  TmpArrBlock->setName(NewLoopHeader->getName() + Twine(".tmparr"));

  Constant *tmpArrSize = ConstantInt::getSigned(NewLoopIndVar->getType(), TripCount);
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
                                        //NewLoopHeader->getFirstNonPHI());

    NumTmpArrays++;

    SmallVector<Value*, 8> GEPIdx;
    GEPIdx.push_back(NewLoopIndVar);
    GetElementPtrInst *GEP = GetElementPtrInst::Create(tmpArr, GEPIdx, Instr->getName() + Twine(".idx"),
                              NewLoopHeader->getFirstNonPHI());

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

static bool hoist(Instruction *I, BasicBlock *InsertionBlock) {
  I->moveBefore(InsertionBlock->getTerminator());
  NumHoisted++;
  return true;
}

static void replaceIndVarOperand(Instruction *I, PHINode *Old, PHINode *New) {
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (PHINode *PhiOp = dyn_cast<PHINode>(I->getOperand(i)))
      if (PhiOp == Old)
        I->replaceUsesOfWith(Old, New);
}

static bool inSubLoop(BasicBlock *BB, Loop *CurLoop, LoopInfo *LI) {
  assert(CurLoop->contains(BB) && "Only valid if BB is IN the loop");
  return LI->getLoopFor(BB) != CurLoop;
}
