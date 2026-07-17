.code

; Windows API switchToFiber() bloats the context switch to roughly 50 to 100 nanoseconds b/c it does a lot more than swap the stack pointer.
; We will write our own custom fiber context switch by savig the exact hardware registers abnd physically swap the RSP (Stack Pointer) register, takes ~3 nanoseconds.

; extern "C" void* SwapContext(void** current_rsp, void* target_rsp);
; RCX = address to store the current thread's RSP
; RDX = the target fiber's RSP to load into the CPU

SwapContext PROC
    ; 1. Save Callee-Saved Integer Registers
    push rbp
    push rbx
    push rdi
    push rsi
    push r12
    push r13
    push r14
    push r15

    ; 2. Save Callee-Saved SIMD Registers (XMM6 through XMM15)
    ; We allocate 160 bytes on the stack (10 registers * 16 bytes)
    sub rsp, 160
    movaps [rsp+144], xmm15
    movaps [rsp+128], xmm14
    movaps [rsp+112], xmm13
    movaps [rsp+96],  xmm12
    movaps [rsp+80],  xmm11
    movaps [rsp+64],  xmm10
    movaps [rsp+48],  xmm9
    movaps [rsp+32],  xmm8
    movaps [rsp+16],  xmm7
    movaps [rsp],     xmm6

    ; ==========================================
    ; THE HARDWARE CONTEXT SWITCH
    ; ==========================================
    
    ; 3. Save current Stack Pointer (RSP) into the address pointed to by RCX
    mov [rcx], rsp

    ; 4. Load the target Stack Pointer (RSP) from RDX
    mov rsp, rdx

    ; ==========================================
    ; WE ARE NOW ON THE NEW FIBER'S STACK
    ; ==========================================

    ; 5. Restore Callee-Saved SIMD Registers
    movaps xmm6,  [rsp]
    movaps xmm7,  [rsp+16]
    movaps xmm8,  [rsp+32]
    movaps xmm9,  [rsp+48]
    movaps xmm10, [rsp+64]
    movaps xmm11, [rsp+80]
    movaps xmm12, [rsp+96]
    movaps xmm13, [rsp+112]
    movaps xmm14, [rsp+128]
    movaps xmm15, [rsp+144]
    add rsp, 160

    ; 6. Restore Callee-Saved Integer Registers
    pop r15
    pop r14
    pop r13
    pop r12
    pop rsi
    pop rdi
    pop rbx
    pop rbp

    ; 7. Return to whoever called SwapContext, but on the NEW stack!
    ret

SwapContext ENDP
END
