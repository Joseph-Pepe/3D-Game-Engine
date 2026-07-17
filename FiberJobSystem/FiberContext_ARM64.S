.global SwapContext
.align 2

/*
 * Apple Silicon, Nintendo Switch 2, Android
 * x0 = current_rsp
 * x1 = target_rsp
 */
SwapContext:
    /* 1. Save Callee-Saved Integer Registers + FP/LR (96 bytes) */
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!
    stp x29, x30, [sp, #-16]!  // x30 is the Link Register (Return Address)

    /* 2. Save Callee-Saved SIMD Registers (64 bytes) */
    stp d8,  d9,  [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    /* 3. Swap the stack pointers */
    mov x2, sp
    str x2, [x0]
    mov sp, x1

    /* 4. Restore SIMD Registers */
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8,  d9,  [sp], #16

    /* 5. Restore Integer Registers */
    ldp x29, x30, [sp], #16
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16

    ret
