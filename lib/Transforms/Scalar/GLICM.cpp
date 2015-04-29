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
static bool isUsedOutsideOfLoop(Instruction *I, Loop *L, LoopInfo *LI);
static bool loopMayThrow(Loop *L, Loop *CurLoop, LoopInfo *LI);

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
      AU.addRequiredID(LCSSAID);
      AU.addRequiredID(LoopSimplifyID);
    }

  private:
    // TODO: Add documentation for fields and functions.
    int counter = 0;
    long InstrIndex = 0;

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
    void replaceUsesWithValueInArray(Instruction *I, AllocaInst *Arr, Value *Index);
    AllocaInst *createTemporaryArray(Instruction *I, Constant *Size,
                                     BasicBlock* BB);
    void storeInstructionInArray(Instruction *I, AllocaInst *Arr, Value *Index);
    bool isGLICMProfitable(LICMSafetyInfo *SafetyInfo);
  };
}

char GLICM::ID = 0;
INITIALIZE_PASS_BEGIN(GLICM, "glicm", "GLICM", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(GLICM, "glicm", "GLICM", false, false)

Pass *llvm::createGLICMPass() { return new GLICM(); }

bool GLICM::runOnLoop(Loop *L, LPPassManager &LPM) {
  bool Changed = false;
  ScalarEvolution *SCEV = &getAnalysis<ScalarEvolution>();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  AA = &getAnalysis<AliasAnalysis>();

  CurLoop = L;
  unsigned TripCount = SCEV->getSmallConstantTripCount(L);
  IndVar = L->getCanonicalInductionVariable();
  ParentLoop = L->getParentLoop();
  bool ParentLoopMayThrow = false;

  if (ParentLoop)
    ParentLoopMayThrow = loopMayThrow(ParentLoop, CurLoop, LI);

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

  // Add BB's from the parent loop to the alias set tracker.
  if (ParentLoop) {
    for (Loop::block_iterator I = ParentLoop->block_begin(),
         E = ParentLoop->block_end(); I != E; ++I) {
      BasicBlock *BB = *I;
      if (LI->getLoopFor(BB) != L)
        CurAST->add(*BB);
    }
    LoopToAliasSetMap[L] = CurAST;
  } else
    delete CurAST;

  // Apply generalized loop-invariant code motion to loops that satisfy all the
  // following conditions:
  // 1) they have a canonical induction variable (starting from 0 and
  //    incremented by 1 each iteration) and a computable trip count. This is
  //    needed in order to clone the structure of the loop in the parent loop's
  //    pre-header.
  // 2) they have a parent loop which is in loop simplify form (to ensure the
  //    loop has a valid pre-header)
  if (!(ParentLoop && ParentLoop->isLoopSimplifyForm() && IndVar &&
      TripCount > 0 && !ParentLoopMayThrow)) {
    return Changed;
  }

  // Compute Loop safety information for CurLoop.
  LICMSafetyInfo CurLoopSafetyInfo;
  computeLICMSafetyInfo(&CurLoopSafetyInfo, CurLoop);

  // Clone the current loop in the preheader of its parent loop.
  createMirrorLoop(ParentLoop, TripCount);

  // Move appropriate instructions from the original loop to the cloned loop.
  generalizedHoist(DT->getNode(L->getHeader()), &CurLoopSafetyInfo);

  // Replace uses of the hoisted instructions in the original loop with uses of
  // results loaded from temporary arrays.
  replaceScalarsWithArrays(TripCount);
  return true;
}

// Creates an empty loop with the same iteration space as CurLoop in the
// pre-header of loop L.
void GLICM::createMirrorLoop(Loop *L, unsigned TripCount) {
  BasicBlock *Header = L->getHeader();
  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Header || !Preheader)
    return;

  // Create a header and a latch for the new loop.
  NewLoopHeader = SplitEdge(Preheader, Header, DT, LI);
  NewLoopLatch = SplitEdge(NewLoopHeader, Header, DT, LI);

  StringRef CurLoopRootName = CurLoop->getHeader()->getName();
  NewLoopHeader->setName(CurLoopRootName + Twine(".gcm.1"));
  NewLoopLatch->setName(CurLoopRootName + Twine(".gcm.2"));
  NewLoopPreheader = NewLoopHeader->getUniquePredecessor();

  // Add a new PHINode at the end of NewLoopHeader. This node corresponds to
  // the induction variable of the new (cloned) loop.
  Twine NewLoopIndVarName = IndVar->getName() + Twine(".gcm");
  NewLoopIndVar = PHINode::Create(IndVar->getType(), 2, NewLoopIndVarName,
                                  NewLoopHeader->getTerminator());

  // Increment the induction variable by 1. Insert the instruction at the end of
  // the latch BasicBlock.
  BinaryOperator *NextIndVar =
      BinaryOperator::CreateAdd(NewLoopIndVar,
                                ConstantInt::getSigned(NewLoopIndVar->getType(),
                                                       1),
                                NewLoopIndVarName + Twine(".inc"),
                                NewLoopLatch->getTerminator());

  // Insert a cmp instruction between the induction variable and the known trip
  // count. Insert the instruction at the end of the latch BasicBlock.
  CmpInst *CmpInst =
      CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_SLT, NextIndVar,
                      ConstantInt::getSigned(NextIndVar->getType(), TripCount),
                      NewLoopIndVarName + Twine(".cmp"),
                      NewLoopLatch->getTerminator());

  // At the moment, the cloned loop is not complete. We need to replace the
  // unconditional branch from NewLoopLatch to Header with a conditional branch,
  // that proceeds into Header only when the cloned loop finishes executing.
  //
  // If we simply remove the unconditional branch at the end of NewLoopLatch,
  // any PHINodes in Header whose values depend on entry from NewLoopLatch will
  // be removed. Thus we create an empty basic block between NewLoopLatch and
  // Header. Now we can replace the branch.
  BasicBlock *Dummy = SplitEdge(NewLoopLatch, Header, DT, LI);
  Dummy->setName(CurLoopRootName + Twine(".gcm.end"));
  Dummy->removePredecessor(NewLoopLatch);
  NewLoopLatch->getTerminator()->eraseFromParent();

  // Create a conditional branch based on CmpInst.
  BranchInst::Create(NewLoopHeader, Dummy, CmpInst, NewLoopLatch);

  // Finally, fill in incoming values for the PHINode of the new induction
  // variable:
  // 1) if control arrives from NewLoopPreheader, the loop is at its first
  //    iteration, so the induction variable is 0.
  // 2) if control arrives from NewLoopLatch, the induction variable must be
  //    updated to the value of NextIndVar.
  NewLoopIndVar->addIncoming(ConstantInt::getSigned(NewLoopIndVar->getType(),
                                                    0),
                             NewLoopPreheader);
  NewLoopIndVar->addIncoming(NextIndVar, NewLoopLatch);
}

bool GLICM::generalizedHoist(DomTreeNode* N, LICMSafetyInfo *SafetyInfo) {
  bool Changed = false;
  BasicBlock *BB = N->getBlock();

  if (!CurLoop->contains(BB))
    return Changed;
  if (CurLoop->getLoopLatch() == BB)
    return Changed;

  if (!inSubLoop(BB, CurLoop, LI))
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E; ) {
      Instruction &I = *II++;
      if (&I == IndVar)
        // This is the induction variable, do not hoist it.
        continue;

      // dbgs() << "Instruction: " << I << "\n";
      // bool i1 = hasLoopInvariantOperands(&I, ParentLoop);
      // bool i2 = canSinkOrHoistInst(I, AA, DT, CurLoop, CurAST, SafetyInfo);
      // bool i3 = isSafeToExecuteUnconditionally(I, DT, CurLoop, SafetyInfo);
      // dbgs() << i1 << " " << i2 << " " << i3 << "\n";
      // if (i1 && i2 && i3) {
      //dbgs() << "[Can be hoisted = " << canSinkOrHoistInst(I, AA, DT, CurLoop, CurAST, SafetyInfo) <<
        //      "] : Instruction: " << I << "\n";
      if (hasLoopInvariantOperands(&I, ParentLoop) &&
          canSinkOrHoistInst(I, AA, DT, CurLoop, CurAST, SafetyInfo) &&
          isSafeToExecuteUnconditionally(I, DT, CurLoop, SafetyInfo) &&
          !isUsedOutsideOfLoop(&I, CurLoop, LI)) {
        // dbgs() << "GLICM is hoisting the following instruction: " << I << "\n";
        // dbgs() << "The loop canonical indvar is: " << *IndVar << "\n";
        // dbgs() << "The loop is: \n" << *CurLoop << "\n";
        // dbgs() << "The current BB: " << *BB << "\n";
        // dbgs() << "*****************************************************\n";
        Changed |= hoist(&I, NewLoopHeader);
        replaceIndVarOperand(&I, IndVar, NewLoopIndVar);
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

    // If this operand is defined in the parent loop, it is clearly not
    // invariant with respect to it.
    if (LI->getLoopFor(I->getParent()) == OuterLoop)
      return false;

    // If this operand is the canonical induction variable of the current loop,
    // we can hoist it and replace it with the induction variable in the cloned
    // loop.
    if (PHINode *PhiNode = dyn_cast<PHINode>(I))
      if (PhiNode == IndVar)
        return true;
    return CurLoop->isLoopInvariant(I);
  }

  // Non-instructions are loop-invariant by default.
  return true;
}

void GLICM::replaceScalarsWithArrays(unsigned TripCount) {

  std::vector<Instruction*> InstrUsedOutsideLoop;
  // Construct a list of all the instructions in the cloned loop that are used
  // in the original. These will need to be stored in temporary arrays and their
  // uses replaced.
  for (BasicBlock::iterator II = NewLoopHeader->begin(),
       E = NewLoopHeader->end(); II != E; ) {
    Instruction &I = *II++;

    // Skip the PHINode defining the canonical induction variable of the cloned
    // loop.
    if (&I == NewLoopIndVar)
      continue;

    if (isUsedInOriginalLoop(&I))
      InstrUsedOutsideLoop.push_back(&I);
  }

  if (InstrUsedOutsideLoop.empty())
    return;

  // Create a BasicBlock for temporary array definitions.
  BasicBlock *TmpArrBlock = SplitEdge(NewLoopPreheader, NewLoopHeader, DT, LI);
  TmpArrBlock->setName(NewLoopHeader->getName() + Twine(".tmparr"));
  Constant *TmpArrSize = ConstantInt::getSigned(NewLoopIndVar->getType(),
                                                TripCount);

  for (std::vector<Instruction*>::iterator II = InstrUsedOutsideLoop.begin();
       II != InstrUsedOutsideLoop.end(); ) {
    Instruction *CurInstr = *II++;
    AllocaInst *TmpArr = createTemporaryArray(CurInstr, TmpArrSize,
                                              TmpArrBlock);
    storeInstructionInArray(CurInstr, TmpArr, NewLoopIndVar);
    replaceUsesWithValueInArray(CurInstr, TmpArr, IndVar);
  }

  InstrUsedOutsideLoop.clear();
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

// Replaces uses of I in CurLoop with Arr[Index].
void GLICM::replaceUsesWithValueInArray(Instruction *I, AllocaInst *Arr,
                                        Value *Index) {
    SmallVector<Value*, 8> GEPIndices;
    GEPIndices.push_back(Index);

    BasicBlock *BB = CurLoop->getHeader();
    // Insert a GEP instruction at the beginning of BB.
    GetElementPtrInst *GEP =
        GetElementPtrInst::Create(Arr->getAllocatedType(), Arr, GEPIndices,
                                  "glicm.arrayidx." + Twine(InstrIndex++),
                                  BB->getFirstNonPHI());

    // Load Arr[Index], insert it at the beginning of BB (but after GEP).
    LoadInst *Load = new LoadInst(GEP,
                                  "glicm.load." + Twine(InstrIndex++),
                                  BB->getFirstNonPHI());
    Load->removeFromParent();
    Load->insertAfter(GEP);

    // Replace all uses of I in BB with Arr[Index].
    I->replaceUsesOutsideBlock(Load, NewLoopHeader);
}

// Allocates an array of size Size at the end of BB.
AllocaInst *GLICM::createTemporaryArray(Instruction *I, Constant *Size,
                                        BasicBlock* BB) {
  AllocaInst *TmpArr = new AllocaInst(I->getType(), Size,
                                      "glicm.arr." + Twine(InstrIndex++),
                                      BB->getTerminator());
  NumTmpArrays++;
  return TmpArr;
}

// Stores I in Arr at index Index.
void GLICM::storeInstructionInArray(Instruction *I, AllocaInst *Arr,
                                    Value *Index) {
  SmallVector<Value*, 8> GEPIndices;
  GEPIndices.push_back(Index);

  // First compute the address at which to store I using a GEP instruction.
  GetElementPtrInst *GEP =
      GetElementPtrInst::Create(Arr->getAllocatedType(), Arr, GEPIndices,
                                "glicm.arrayidx." + Twine(InstrIndex++), I);

  // Workaround: we need to specify a location for S when we create it, but the
  // only options are either (a) at the end of BB or (b) before an instruction.
  // We actually want this to be after I. Thus we insert it at the end of the
  // BB, remove it, and then insert it in the proper place.
  StoreInst *S = new StoreInst(I, GEP, I->getParent());
  S->removeFromParent();
  S->insertAfter(I);
}

bool GLICM::isGLICMProfitable(LICMSafetyInfo *SafetyInfo) {
  // bool Profitable = true;

  std::set<Instruction*> HoistableInstr;
  for (Loop::block_iterator BB = CurLoop->block_begin(),
       BBE = CurLoop->block_end(); (BB != BBE); ++BB)
    for (BasicBlock::iterator I = (*BB)->begin(), E = (*BB)->end(); (I != E);
         ++I) {
      Instruction &Instr = *I;
      if ((hasLoopInvariantOperands(&Instr, ParentLoop)) &&
//            (!HoistableInstr.empty() &&
//             !hasLoopInvariantOperands(&Instr, ParentLoop) &&
//             HoistableInstr.find(&Instr) != HoistableInstr.end())) &&
          canSinkOrHoistInst(Instr, AA, DT, CurLoop, CurAST, SafetyInfo) &&
          isSafeToExecuteUnconditionally(Instr, DT, CurLoop, SafetyInfo) &&
          !isUsedOutsideOfLoop(&Instr, CurLoop, LI)) {
        dbgs() << "Adding " << Instr << "\n";
        HoistableInstr.insert(&Instr);
      }
    }
  dbgs() << "Size of hoistable instruction set: " << HoistableInstr.size() << "\n";
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

static bool isUsedOutsideOfLoop(Instruction *I, Loop *L, LoopInfo *LI) {
  for (User *U : I->users())
    if (Instruction *UserI = dyn_cast<Instruction>(U))
      if (LI->getLoopFor(UserI->getParent()) != L)
        return true;
  return false;
}

static bool loopMayThrow(Loop *L, Loop *CurLoop, LoopInfo *LI) {
  bool MayThrow = false;

  // Check if any instruction within L which does not belong to CurLoop may
  // throw. If yes, then we cannot hoist instructions from CurLoop to L's
  // preheader, because exceptions thrown by instructions of L may prevent
  // that instruction from executing.

  // This is a coarse-grained check, similar to what LICM does in the
  // computeLICMSafetyInfo function. More refined checks are possible, such as
  // checking whether the throwing instructions appear before or after the
  // current loop. With this conservative approach, if there are any
  // instructions that may throw in the L loop, this function will return false.
  for (Loop::block_iterator BB = L->block_begin(),
       BBE = L->block_end(); (BB != BBE) && !MayThrow ; ++BB)
    if (LI->getLoopFor(*BB) != CurLoop)
      for (BasicBlock::iterator I = (*BB)->begin(), E = (*BB)->end();
           (I != E) && !MayThrow; ++I) {
        if (I->mayThrow())
          dbgs() << "Throwing instr: " << *I << "\n";
        MayThrow |= I->mayThrow();
      }

  return MayThrow;
}
