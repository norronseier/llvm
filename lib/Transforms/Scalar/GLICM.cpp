#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

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
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequired<ScalarEvolution>();
      AU.addRequiredID(LoopSimplifyID);
    }

  private:
    int counter = 0;
    Loop *CurLoop;
    Loop *ParentLoop;
    DominatorTree *DT;
    LoopInfo *LI;
    PHINode *IndVar;
    PHINode *phiNode;
    BasicBlock *NewLoopBody;
    BasicBlock *NewLoopCond;

    void createMirrorLoop(Loop *L, unsigned TripCount);
    bool generalizedHoist(DomTreeNode *N);
    bool isInvariantInGivenLoop(Instruction *I, Loop *L);
    void hoist(Instruction *I);
  };
}

char GLICM::ID = 0;
INITIALIZE_PASS_BEGIN(GLICM, "glicm", "GLICM", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_END(GLICM, "glicm", "GLICM", false, false)

bool GLICM::runOnLoop(Loop *L, LPPassManager &LPM) {
  ScalarEvolution *SCEV = &getAnalysis<ScalarEvolution>();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  dbgs() << "===== Current loop: " << *L << " ========\n";
  CurLoop = L;
  unsigned TripCount = SCEV->getSmallConstantTripCount(L);
  IndVar = L->getCanonicalInductionVariable();
  ParentLoop = L->getParentLoop();

  // Apply this to loops that have a canonical indvar (starting from 0 and incrementing by 1
  // on each loop step) and a statically computable trip count.
  if (ParentLoop && ParentLoop->isLoopSimplifyForm() &&
      IndVar && TripCount > 0) {
    createMirrorLoop(ParentLoop, TripCount);
    generalizedHoist(DT->getNode(L->getHeader()));
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

bool GLICM::generalizedHoist(DomTreeNode* N) {
  bool Changed = false;
  BasicBlock *BB = N->getBlock();

  if (!CurLoop->contains(BB))
    return Changed;

  if (!inSubLoop(BB, CurLoop, LI)) {
    dbgs() << "In BB:" << BB->getName() << "\n";
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E; ) {
      Instruction &I = *II++;
      if (&I == IndVar)
        // This is the induction variable, do not hoist it
        continue;

      dbgs() << "Instruction: " << I << "\n";
      if (isInvariantInGivenLoop(&I, ParentLoop)) {
        hoist(&I);
        break;
      }
    }
  }
  const std::vector<DomTreeNode*> &Children = N->getChildren();
  for (unsigned i = 0, e = Children.size(); i != e; ++i)
    Changed |= generalizedHoist(Children[i]);
  return Changed;
}

bool GLICM::isInvariantInGivenLoop(Instruction *I, Loop *L) {
  if (PHINode *PhiNode = dyn_cast<PHINode>(I))
    if (PhiNode == IndVar) {
      dbgs() << "the canonical ind var\n";
      return true;
    }

  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (Instruction *Instr = dyn_cast<Instruction>(I->getOperand(i)))
      if (LI->getLoopFor(Instr->getParent()) == L)
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

static bool inSubLoop(BasicBlock *BB, Loop *CurLoop, LoopInfo *LI) {
  assert(CurLoop->contains(BB) && "Only valid if BB is IN the loop");
  return LI->getLoopFor(BB) != CurLoop;
}
