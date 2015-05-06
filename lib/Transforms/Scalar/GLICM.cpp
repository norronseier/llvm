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
    LICMSafetyInfo *SafetyInfo;

    Loop *CurLoop;
    Loop *ParentLoop;
    PHINode *IndVar;
    PHINode *ClonedLoopIndVar;
    BasicBlock *ClonedLoopHeader;
    BasicBlock *ClonedLoopLatch;
    BasicBlock *ClonedLoopPreheader;

    void createMirrorLoop(Loop *L, unsigned TripCount);
    bool generalizedHoist(DomTreeNode *N);
    bool hasLoopInvariantOperands(Instruction *I, Loop *OuterLoop);
    bool isInvariantForGLICM(Value *V, Loop *OuterLoop);
    void replaceScalarsWithArrays(unsigned TripCount);
    bool isUsedInOriginalLoop(Instruction *I);
    void replaceUsesWithValueInArray(Instruction *I, AllocaInst *Arr,
                                     Value *Index);
    AllocaInst *createTemporaryArray(Instruction *I, Constant *Size,
                                     BasicBlock* BB);
    void storeInstructionInArray(Instruction *I, AllocaInst *Arr, Value *Index);
    bool isProfitableToHoist(Instruction *I);
    bool hasLoopInvariantOperandsApartFrom(Instruction *I, Loop *OuterLoop,
                                           Value *V);
    bool shouldHoist(Instruction *I, Value *V);
  };
}

char GLICM::ID = 0;
INITIALIZE_PASS_BEGIN(GLICM, "glicm", "Generalized Loop-Invariant Code Motion",
                      true, /* Modifies the CFG of the function */
                      false) /* Is not an analysis pass */
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(GLICM, "glicm", "GLICM", true, false)

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
  SafetyInfo = new LICMSafetyInfo();
  computeLICMSafetyInfo(SafetyInfo, CurLoop);

  // Clone the current loop in the preheader of its parent loop.
  createMirrorLoop(ParentLoop, TripCount);

  // Move appropriate instructions from the original loop to the cloned loop.
  Changed = generalizedHoist(DT->getNode(L->getHeader()));

  // Replace uses of the hoisted instructions in the original loop with uses of
  // results loaded from temporary arrays.
  replaceScalarsWithArrays(TripCount);

  delete SafetyInfo;
  return Changed;
}

/// Creates an empty loop with the same iteration space as CurLoop in the
/// pre-header of loop L.
void GLICM::createMirrorLoop(Loop *L, unsigned TripCount) {
  BasicBlock *Header = L->getHeader();
  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Header || !Preheader)
    return;

  // Create a header and a latch for the new loop.
  ClonedLoopHeader = SplitEdge(Preheader, Header, DT, LI);
  ClonedLoopLatch = SplitEdge(ClonedLoopHeader, Header, DT, LI);

  StringRef CurLoopRootName = CurLoop->getHeader()->getName();
  ClonedLoopHeader->setName(CurLoopRootName + Twine(".gcm.1"));
  ClonedLoopLatch->setName(CurLoopRootName + Twine(".gcm.2"));
  ClonedLoopPreheader = ClonedLoopHeader->getUniquePredecessor();

  // Add a new PHINode at the end of ClonedLoopHeader. This node corresponds to
  // the induction variable of the new (cloned) loop.
  Twine ClonedLoopIndVarName = IndVar->getName() + Twine(".gcm");
  ClonedLoopIndVar = PHINode::Create(IndVar->getType(), 2, ClonedLoopIndVarName,
                                  ClonedLoopHeader->getTerminator());

  // Increment the induction variable by 1. Insert the instruction at the end of
  // the latch BasicBlock.
  BinaryOperator *NextIndVar =
      BinaryOperator::CreateAdd(ClonedLoopIndVar,
                                ConstantInt::getSigned(ClonedLoopIndVar->getType(),
                                                       1),
                                ClonedLoopIndVarName + Twine(".inc"),
                                ClonedLoopLatch->getTerminator());

  // Compare the induction variable with the known trip count. Insert the
  // instruction at the end of the latch BasicBlock.
  CmpInst *CmpInst =
      CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_SLT, NextIndVar,
                      ConstantInt::getSigned(NextIndVar->getType(), TripCount),
                      ClonedLoopIndVarName + Twine(".cmp"),
                      ClonedLoopLatch->getTerminator());

  // At the moment, the cloned loop is not complete. We need to replace the
  // unconditional branch from ClonedLoopLatch to Header with a conditional
  // branch, that proceeds into Header only when the cloned loop finishes
  // executing.
  //
  // If we simply remove the unconditional branch at the end of ClonedLoopLatch,
  // any PHINodes in Header whose values depend on entry from ClonedLoopLatch will
  // be removed. Thus we create an empty basic block between ClonedLoopLatch and
  // Header. Now we can replace the branch.
  BasicBlock *Dummy = SplitEdge(ClonedLoopLatch, Header, DT, LI);
  Dummy->setName(CurLoopRootName + Twine(".gcm.end"));
  Dummy->removePredecessor(ClonedLoopLatch);
  ClonedLoopLatch->getTerminator()->eraseFromParent();

  // Create a conditional branch based on CmpInst.
  BranchInst::Create(ClonedLoopHeader, Dummy, CmpInst, ClonedLoopLatch);

  // Finally, fill in incoming values for the PHINode of the new induction
  // variable:
  // 1) if control arrives from ClonedLoopPreheader, the loop is at its first
  //    iteration, so the induction variable is 0.
  // 2) if control arrives from ClonedLoopLatch, the induction variable must be
  //    updated to the value of NextIndVar.
  ClonedLoopIndVar->addIncoming(ConstantInt::getSigned(ClonedLoopIndVar->getType(),
                                                    0),
                             ClonedLoopPreheader);
  ClonedLoopIndVar->addIncoming(NextIndVar, ClonedLoopLatch);
}

/// Applies generalized loop-invariant code motion to the current loop.
bool GLICM::generalizedHoist(DomTreeNode* N) {
  bool Changed = false;
  BasicBlock *BB = N->getBlock();

  if (!CurLoop->contains(BB))
    return Changed;

  if (!inSubLoop(BB, CurLoop, LI))
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E; ) {
      Instruction &I = *II++;
      if (&I == IndVar)
        // This is the PHI node representing the induction variable, so do not
        // hoist it.
        continue;

      if (shouldHoist(&I, NULL)) {
        Changed |= hoist(&I, ClonedLoopHeader);
        replaceIndVarOperand(&I, IndVar, ClonedLoopIndVar);
      }
    }

  const std::vector<DomTreeNode*> &Children = N->getChildren();
  for (unsigned i = 0, e = Children.size(); i != e; ++i)
    Changed |= generalizedHoist(Children[i]);

  return Changed;
}

/// Returns true if I is invariant with respect to OuterLoop.
bool GLICM::hasLoopInvariantOperands(Instruction *I, Loop *OuterLoop) {
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (!isInvariantForGLICM(I->getOperand(i), OuterLoop))
      return false;

  return true;
}

/// Returns true if I has any operands apart from V which are invariant with
/// respect to OuterLoop.
bool GLICM::hasLoopInvariantOperandsApartFrom(Instruction *I, Loop *OuterLoop,
                                              Value *V) {
  for (unsigned i = 0, e = I->getNumOperands(); i != e; i++) {
    Value *Op = I->getOperand(i);
    if (!isInvariantForGLICM(Op, OuterLoop) && Op != V)
      return false;
  }

  return true;
}

/// Returns true if the given value is invariant with respect to OuterLoop.
bool GLICM::isInvariantForGLICM(Value *V, Loop *OuterLoop) {
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    // If this value is defined in the parent loop, it is clearly not
    // invariant with respect to it.
    if (LI->getLoopFor(I->getParent()) == OuterLoop)
      return false;

    // If this value is the PHI node of the canonical induction variable,
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

/// Stores all definitions in the cloned loop who have uses in the original loop
/// into arrays.
void GLICM::replaceScalarsWithArrays(unsigned TripCount) {

  std::vector<Instruction*> InstrUsedOutsideLoop;

  // Construct a list of all the definitions in the cloned loop that are used
  // in the original one.
  for (BasicBlock::iterator II = ClonedLoopHeader->begin(),
       E = ClonedLoopHeader->end(); II != E; ) {
    Instruction &I = *II++;

    // Skip the PHINode defining the canonical induction variable of the cloned
    // loop.
    if (&I == ClonedLoopIndVar)
      continue;

    if (isUsedInOriginalLoop(&I))
      InstrUsedOutsideLoop.push_back(&I);
  }

  if (InstrUsedOutsideLoop.empty())
    return;

  // Create a BasicBlock for array definitions.
  BasicBlock *ArrBlock = SplitEdge(ClonedLoopPreheader, ClonedLoopHeader, DT, LI);
  ArrBlock->setName(ClonedLoopHeader->getName() + Twine(".arr"));
  Constant *ArrSize = ConstantInt::getSigned(ClonedLoopIndVar->getType(),
                                                TripCount);

  for (std::vector<Instruction*>::iterator II = InstrUsedOutsideLoop.begin();
       II != InstrUsedOutsideLoop.end(); ) {
    Instruction *CurInstr = *II++;

    // Define a temporary array for storing the values of CurInstr in each
    // iteration of the cloned loop.
    AllocaInst *Arr = createTemporaryArray(CurInstr, ArrSize, ArrBlock);

    // Store the value of CurInstr at the ith iteration of the cloned loop at
    // Arr[i].
    storeInstructionInArray(CurInstr, Arr, ClonedLoopIndVar);

    // Replace uses of CurInstr at the ith iteration of the original loop with
    // Arr[i].
    replaceUsesWithValueInArray(CurInstr, Arr, IndVar);
  }

  InstrUsedOutsideLoop.clear();
}

/// Returns true if the given definition (present in the cloned loop) is used in
/// the current loop.
bool GLICM::isUsedInOriginalLoop(Instruction *I) {
  assert(I->getParent() == ClonedLoopHeader);
  for (User *U : I->users()) {
    if (Instruction *UserInstr = dyn_cast<Instruction>(U)) {
      if (LI->getLoopFor(UserInstr->getParent()) == CurLoop)
        return true;
    }
  }
  return false;
}

/// Replaces all uses of I in CurLoop with Arr[Index].
void GLICM::replaceUsesWithValueInArray(Instruction *I, AllocaInst *Arr,
                                        Value *Index) {
    SmallVector<Value*, 8> GEPIndices;
    GEPIndices.push_back(Index);

    BasicBlock *BB = CurLoop->getHeader();
    // Create a GEP instruction for computing the address of the element at
    // Arr[Index].
    GetElementPtrInst *GEP =
        GetElementPtrInst::Create(Arr->getAllocatedType(), Arr, GEPIndices,
                                  "glicm.arrayidx." + Twine(InstrIndex++),
                                  BB->getFirstNonPHI());

    // Load Arr[Index].
    LoadInst *Load = new LoadInst(GEP,
                                  "glicm.load." + Twine(InstrIndex++),
                                  BB->getFirstNonPHI());
    Load->removeFromParent();
    Load->insertAfter(GEP);

    // Replace all uses of I inside the original loop with Arr[Index]. Since all
    // uses of I apart from those in the cloned loop are guaranteed to be in
    // the current loop (see conditions for hoisting), this is equivalent to the
    // call below.
    I->replaceUsesOutsideBlock(Load, ClonedLoopHeader);
}

/// Allocates an array holding Size elements of the same type as I at the end of
/// the given BB.
AllocaInst *GLICM::createTemporaryArray(Instruction *I, Constant *Size,
                                        BasicBlock* BB) {
  AllocaInst *TmpArr = new AllocaInst(I->getType(), Size,
                                      "glicm.arr." + Twine(InstrIndex++),
                                      BB->getTerminator());
  NumTmpArrays++;
  return TmpArr;
}

/// Creates a StoreInst that stores the result of I in Arr[Index] for the given
/// array and index, and inserts it after I.
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

/// Returns true when it is profitable to hoist the given instruction to the
/// cloned loop.
bool GLICM::isProfitableToHoist(Instruction *Instr) {

  // We hoist GEP and Load instructions only if at least one of their users can
  // be subsequently hoisted. This prevents pathological scenarios such as:
  // 1) A GEP instruction is hoisted to the cloned loop, its computed address is
  //    stored in a temporary array, and then loaded from that array in the
  //    original loop (incurring an additional GEP and load).
  // 2) A load instruction is hoisted to the cloned loop, its value is stored in
  //    a temporary array, and then loaded from that array in the original loop
  //    (incurring an additional GEP and load).
  if (isa<GetElementPtrInst>(Instr) || isa<LoadInst>(Instr)) {
    for (User *U: Instr->users())
      if (Instruction *I = dyn_cast<Instruction>(U))
        if (shouldHoist(I, Instr))
          return true;
  }

  // If this instruction is an incoming value for the induction variable's PHI
  // node, don't hoist it out, since it does not add benefits.
  for (User *U : Instr->users())
    if (Instruction *I = dyn_cast<Instruction>(U)) {
      if (I == IndVar)
        return false;
    }

  // Hoist any binary operation, since this is the type of instruction we are
  // usually most interested in.
  if (Instr->isBinaryOp())
    return true;

  // Otherwise do not hoist this instruction.
  return false;
}

/// Returns true if the given instruction should be hoisted out of its loop
/// according to a simple profitability model.
bool GLICM::shouldHoist(Instruction *I, Value *V) {
  return (V == NULL ? hasLoopInvariantOperands(I, ParentLoop)
                    : hasLoopInvariantOperandsApartFrom(I, ParentLoop, V)) &&
         canSinkOrHoistInst(*I, AA, DT, CurLoop, CurAST, SafetyInfo) &&
         isSafeToExecuteUnconditionally(*I, DT, CurLoop, SafetyInfo) &&
         !isUsedOutsideOfLoop(I, CurLoop, LI) &&
         isProfitableToHoist(I);
}

/// Moves I at the end of InsertionBlock.
static bool hoist(Instruction *I, BasicBlock *InsertionBlock) {
  I->moveBefore(InsertionBlock->getTerminator());
  NumHoisted++;
  return true;
}

/// Replaces all uses of Old with New amongst I's operands.
static void replaceIndVarOperand(Instruction *I, PHINode *Old, PHINode *New) {
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (PHINode *PhiOp = dyn_cast<PHINode>(I->getOperand(i)))
      if (PhiOp == Old)
        I->replaceUsesOfWith(Old, New);
}

/// Returns true if BB is part of CurLoop.
static bool inSubLoop(BasicBlock *BB, Loop *CurLoop, LoopInfo *LI) {
  assert(CurLoop->contains(BB) && "Only valid if BB is IN the loop");
  return LI->getLoopFor(BB) != CurLoop;
}

/// Returns true if any uses of I lie outside of the loop L.
static bool isUsedOutsideOfLoop(Instruction *I, Loop *L, LoopInfo *LI) {
  for (User *U : I->users())
    if (Instruction *UserI = dyn_cast<Instruction>(U))
      if (LI->getLoopFor(UserI->getParent()) != L)
        return true;
  return false;
}

/// Returns true if any instruction in OuterLoop which is not part of CurLoop
/// may throw an exception (assumes CurLoop is a subloop of InnerLoop).
static bool loopMayThrow(Loop *OuterLoop, Loop *CurLoop, LoopInfo *LI) {
  bool MayThrow = false;

  // Check if any instruction within OuterLoop which does not belong to CurLoop
  // may throw.

  // This is a coarse-grained check, similar to what LICM does in the
  // computeLICMSafetyInfo function. More refined checks are possible, such as
  // checking whether the throwing instructions appear before or after the
  // current loop.
  for (Loop::block_iterator BB = OuterLoop->block_begin(),
       BBE = OuterLoop->block_end(); (BB != BBE) && !MayThrow ; ++BB)
    if (LI->getLoopFor(*BB) != CurLoop)
      for (BasicBlock::iterator I = (*BB)->begin(), E = (*BB)->end();
           (I != E) && !MayThrow; ++I) {
        MayThrow |= I->mayThrow();
      }
  return MayThrow;
}
