.global SwapContext
.type SwapContext, @function

/*
 * Linux, PS5
 * rdi = current_rsp
 * rsi = target_rsp
*/
SwapContext:
    /* 1. Save Callee-Saved Integer Registers (48 bytes) */
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15

    /* 2. Swap the stack pointers */
    movq %rsp, (%rdi)
    movq %rsi, %rsp

    /* 3. Restore Callee-Saved Integer Registers */
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp

    ret
