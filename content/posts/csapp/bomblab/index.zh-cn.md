---
title: "Bomblab"
date: 2023-02-03T14:57:35+08:00
categories: ["csapp"]
summary: "Bomblab from CSAPP 15213, including 6 normal phases and 1 extra secret-phase. It's a little bit hard, please be patient and gradually find your way out, good luck! Source: [https://github.com/yewentao256/CSAPP_15213/tree/main/bomblab]"
---

## Summary

Bomblab from CSAPP 15213, including 6 normal phases and 1 extra secret-phase. It's a little bit hard, please be patient and gradually find your way out, good luck! Source: [https://github.com/yewentao256/CSAPP_15213/tree/main/bomblab]

## 等待被翻译

非常抱歉，看起来这篇博文还没有被翻译成中文，请等待一段时间

## Set up Environment

Using a docker container is the simplest way, source from yansongsongsong

`docker run --privileged -d -p 1221:22 --name bomb yansongsongsong/csapp:bomblab`

Then using vscode `remote ssh` to connect with it as we described in datalab

password: `THEPASSWORDYOUCREATED`

## Commands We usually use

```bash
gdb -q bomb      # start debugging
b explode_bomb   # help you break before bomb
stepi            # run into next instruction (stepin)
nexti            # run into next instruction (not stepin the funcs)
disas phase_1    # make binary coding into assembly, helpful
x/s 0x402400     # get the string value in address 0x402400
i registers      # print the register infos 
p $rsp           # print the value of variable
```

## Phase_1

assembler code for function phase_1:

```c
0x000400ee0 <+0>:     sub    $0x8,%rsp
0x000400ee4 <+4>:     mov    $0x402400,%esi         // move 0x402400 to %esi
0x000400ee9 <+9>:     callq  0x401338 <strings_not_equal>  
0x000400eee <+14>:    test   %eax,%eax              // judge if eax == 1
0x000400ef0 <+16>:    je     0x400ef7 <phase_1+23>  // jump if equal/zero
0x000400ef2 <+18>:    callq  0x40143a <explode_bomb>
0x000400ef7 <+23>:    add    $0x8,%rsp
0x000400efb <+27>:    retq
```

`strings_not_equal` compares two strings in register `%rdi`, `%rsi`, then saves 0 in `%rax` if they are same, 1 otherwise. If you are interested, `disas strings_not_equal` for more details.

So this phase is to compare the strings, if they are not the same, bomb.

So we can use `x/s 0x402400` to see the string, that is the answer.

```bash
(gdb) x/s 0x402400
0x402400:       "Border relations with Canada have never been better."
```

## Phase_2

Dump of assembler code for function phase_2:

```c
0x00400efc <+0>:     push   %rbp
0x00400efd <+1>:     push   %rbx
0x00400efe <+2>:     sub    $0x28,%rsp
0x00400f02 <+6>:     mov    %rsp,%rsi
0x00400f05 <+9>:     callq  0x40145c <read_six_numbers>
// considering read_six_numbers a black box
// after calling this, we find that (%rsp) is the first element we input
// so here we know the first element must be number 1

// what's more, by using `x/8w $rsp` we know the number's relation with rsp
// eg: we input 1 2 3 4 5 6, so 
// x/8w $rsp
// 0x7fffffffe1d0: 0x00000001      0x00000002      0x00000003      0x00000004
// 0x7fffffffe1e0: 0x00000005      0x00000006      0x00401431      0x
0x00400f0a <+14>:    cmpl   $0x1,(%rsp)
0x00400f0e <+18>:    je     0x400f30 <phase_2+52>  // jump if (%rsp) == 1
0x00400f10 <+20>:    callq  0x40143a <explode_bomb>
----------------------------------------------------------
0x00400f15 <+25>:    jmp    0x400f30 <phase_2+52>
0x00400f17 <+27>:    mov    -0x4(%rbx),%eax
0x00400f1a <+30>:    add    %eax,%eax
0x00400f1c <+32>:    cmp    %eax,(%rbx)
// here (%rbx) should equal to the 2 * -0x4(%rbx)
// rbx = rsp + 0x4 when first here
// then(second, third ...), rbx = rbx + 0x4
0x00400f1e <+34>:    je     0x400f25 <phase_2+41> 
0x00400f20 <+36>:    callq  0x40143a <explode_bomb>
0x00400f25 <+41>:    add    $0x4,%rbx
0x00400f29 <+45>:    cmp    %rbp,%rbx
// here is a loop, if rbx == rbp(rsp + 0x18), then quit
// else go to the phase_2 + 27 again
0x00400f2c <+48>:    jne    0x400f17 <phase_2+27>
0x00400f2e <+50>:    jmp    0x400f3c <phase_2+64>
0x00400f30 <+52>:    lea    0x4(%rsp),%rbx
0x00400f35 <+57>:    lea    0x18(%rsp),%rbp
0x00400f3a <+62>:    jmp    0x400f17 <phase_2+27>
0x00400f3c <+64>:    add    $0x28,%rsp
0x00400f40 <+68>:    pop    %rbx
0x00400f41 <+69>:    pop    %rbp
0x00400f42 <+70>:    retq
```

Dump of assembler code for function `read_six_numbers`:

```c
0x0040145c <+0>:     sub    $0x18,%rsp
0x00401460 <+4>:     mov    %rsi,%rdx
0x00401463 <+7>:     lea    0x4(%rsi),%rcx
0x00401467 <+11>:    lea    0x14(%rsi),%rax
0x0040146b <+15>:    mov    %rax,0x8(%rsp)
0x00401470 <+20>:    lea    0x10(%rsi),%rax
0x00401474 <+24>:    mov    %rax,(%rsp)
0x00401478 <+28>:    lea    0xc(%rsi),%r9
0x0040147c <+32>:    lea    0x8(%rsi),%r8
0x00401480 <+36>:    mov    $0x4025c3,%esi
0x00401485 <+41>:    mov    $0x0,%eax
0x0040148a <+46>:    callq  0x400bf0 <__isoc99_sscanf@plt>
// Consider scanf is a black box
// after calling scanf, we find that eax = the number of element we input
// so here we know we should input more than 5 numbers
0x0040148f <+51>:    cmp    $0x5,%eax
0x00401492 <+54>:    jg     0x401499 <read_six_numbers+61>  // jump if eax > 5
0x00401494 <+56>:    callq  0x40143a <explode_bomb>
0x00401499 <+61>:    add    $0x18,%rsp
0x0040149d <+65>:    retq
```

Carefully read all of the codes above, we can know:

- we should input more than five numbers
- the first number should be 1
- numbers[i+i] = numbers[i] * 2

So we get the answer: `1 2 4 8 16 32`

You can also type lots of numbers, that doesn't matter: `1 2 4 8 16 32 64 ...`

## Phase_3

Dump of assembler code for function phase_3:

```c
0x00400f43 <+0>:     sub    $0x18,%rsp
0x00400f47 <+4>:     lea    0xc(%rsp),%rcx
0x00400f4c <+9>:     lea    0x8(%rsp),%rdx
// move $0x4025cf to the second argument of sscanf
// by `x/s 0x4025cf` we get "%d %d", so here we know we should input 2 numbers
0x00400f51 <+14>:    mov    $0x4025cf,%esi
0x00400f56 <+19>:    mov    $0x0,%eax
0x00400f5b <+24>:    callq  0x400bf0 <__isoc99_sscanf@plt>
// according to phase_2, we know %eax is the number of elements we input
// so we should type more than 1 element, which also validates the %d %d above
// what's more, if you command `x/4w $rsp`, you'll find the elements you input
// at 0x8(%rsp), 0xc(%rsp)
0x00400f60 <+29>:    cmp    $0x1,%eax
0x00400f63 <+32>:    jg     0x400f6a <phase_3+39>
0x00400f65 <+34>:    callq  0x40143a <explode_bomb>
// here the first element should not big than 7
0x00400f6a <+39>:    cmpl   $0x7,0x8(%rsp)
0x00400f6f <+44>:    ja     0x400fad <phase_3+106>
---------------------------------------------------------------
0x00400f71 <+46>:    mov    0x8(%rsp),%eax
// calculate address = 8 * rax + 0x402470, then get the value saved in address
// then jump to the value address, usually used in switch table
// 8 means 8 bytes a unit, so we can use `x/8xg 0x402470` to see the table
// 0x402470:       0x00400f7c      0x00400fb9
// 0x402480:       0x00400f83      0x00400f8a
// 0x402490:       0x00400f91      0x00400f98
// 0x4024a0:       0x00400f9f      0x00400fa6
// so we know the first number we input is used to get to the different branch
0x00400f75 <+50>:    jmpq   *0x402470(,%rax,8)
0x00400f7c <+57>:    mov    $0xcf,%eax      // eax = 207
0x00400f81 <+62>:    jmp    0x400fbe <phase_3+123>
0x00400f83 <+64>:    mov    $0x2c3,%eax     // eax = 707
0x00400f88 <+69>:    jmp    0x400fbe <phase_3+123>
0x00400f8a <+71>:    mov    $0x100,%eax     // eax = 256
0x00400f8f <+76>:    jmp    0x400fbe <phase_3+123>
0x00400f91 <+78>:    mov    $0x185,%eax     // eax = 389
0x00400f96 <+83>:    jmp    0x400fbe <phase_3+123>
0x00400f98 <+85>:    mov    $0xce,%eax      // eax = 206
0x00400f9d <+90>:    jmp    0x400fbe <phase_3+123>
0x00400f9f <+92>:    mov    $0x2aa,%eax     // eax = 682
0x00400fa4 <+97>:    jmp    0x400fbe <phase_3+123>
0x00400fa6 <+99>:    mov    $0x147,%eax     // eax = 327
0x00400fab <+104>:   jmp    0x400fbe <phase_3+123>
0x00400fad <+106>:   callq  0x40143a <explode_bomb>
0x00400fb2 <+111>:   mov    $0x0,%eax
0x00400fb7 <+116>:   jmp    0x400fbe <phase_3+123>
0x00400fb9 <+118>:   mov    $0x137,%eax     // eax = 311
// here we compare the second number we input with %eax
// they should be the same
0x00400fbe <+123>:   cmp    0xc(%rsp),%eax
0x00400fc2 <+127>:   je     0x400fc9 <phase_3+134>
0x00400fc4 <+129>:   callq  0x40143a <explode_bomb>
0x00400fc9 <+134>:   add    $0x18,%rsp
0x00400fcd <+138>:   retq
```

Read all of the codes and comments carefully above, we know:

- we should input two numbers
- the first number is used to goto different branches
- the second number should be the same with the value in different branches

So we get the answer, pick one of them:

```c
0 207
1 311
2 707
3 256
4 389
5 206
6 682
7 327
```

## phase_4

Dump of assembler code for function phase_4:

```c
0x0040100c <+0>:     sub    $0x18,%rsp
0x00401010 <+4>:     lea    0xc(%rsp),%rcx
0x00401015 <+9>:     lea    0x8(%rsp),%rdx
0x0040101a <+14>:    mov    $0x4025cf,%esi  // "%d %d", two numbers
0x0040101f <+19>:    mov    $0x0,%eax
0x00401024 <+24>:    callq  0x400bf0 <__isoc99_sscanf@plt>
0x00401029 <+29>:    cmp    $0x2,%eax       // validates two numbers
0x0040102c <+32>:    jne    0x401035 <phase_4+41>
// Note: 0x8(%rsp) is the first number, 0xc(%rsp) is the second
// here the first number(unsigned) should not big than 0xe
0x0040102e <+34>:    cmpl   $0xe,0x8(%rsp)
0x00401033 <+39>:    jbe    0x40103a <phase_4+46>   // below or equal(unsigned)
0x00401035 <+41>:    callq  0x40143a <explode_bomb>
0x0040103a <+46>:    mov    $0xe,%edx   // third arg = 14
0x0040103f <+51>:    mov    $0x0,%esi   // second arg = 0
0x00401044 <+56>:    mov    0x8(%rsp),%edi  // first arg = first n we input
0x00401048 <+60>:    callq  0x400fce <func4>
// here the return number should be 0, or it will boom
0x0040104d <+65>:    test   %eax,%eax
0x0040104f <+67>:    jne    0x401058 <phase_4+76>
// here we know the second number should be zero
0x00401051 <+69>:    cmpl   $0x0,0xc(%rsp)
0x00401056 <+74>:    je     0x40105d <phase_4+81>
0x00401058 <+76>:    callq  0x40143a <explode_bomb>
0x0040105d <+81>:    add    $0x18,%rsp
0x00401061 <+85>:    retq
```

Dump of assembler code for function func4:

```c
0x00400fce <+0>:     sub    $0x8,%rsp
0x00400fd2 <+4>:     mov    %edx,%eax
0x00400fd4 <+6>:     sub    %esi,%eax
0x00400fd6 <+8>:     mov    %eax,%ecx
0x00400fd8 <+10>:    shr    $0x1f,%ecx      // shift logical right 31
0x00400fdb <+13>:    add    %ecx,%eax
0x00400fdd <+15>:    sar    %eax            // shift arithmetic right, default 1
0x00400fdf <+17>:    lea    (%rax,%rsi,1),%ecx
0x00400fe2 <+20>:    cmp    %edi,%ecx
0x00400fe4 <+22>:    jle    0x400ff2 <func4+36>
0x00400fe6 <+24>:    lea    -0x1(%rcx),%edx
0x00400fe9 <+27>:    callq  0x400fce <func4>
0x00400fee <+32>:    add    %eax,%eax
0x00400ff0 <+34>:    jmp    0x401007 <func4+57>
0x00400ff2 <+36>:    mov    $0x0,%eax
0x00400ff7 <+41>:    cmp    %edi,%ecx
0x00400ff9 <+43>:    jge    0x401007 <func4+57>
0x00400ffb <+45>:    lea    0x1(%rcx),%esi
0x00400ffe <+48>:    callq  0x400fce <func4>
0x00401003 <+53>:    lea    0x1(%rax,%rax,1),%eax
0x00401007 <+57>:    add    $0x8,%rsp
0x0040100b <+61>:    retq
```

It's too complicated, so we translate the `func4` to python:

```py
# x = edi = first n we input
# esi = 0 at first, edx = 14 at first
def func4(x: int = 0, esi: int = 0, edx: int = 14) -> int:
    result = edx - esi
    ecx = result >> 31
    result = (result + ecx) >> 1
    ecx = result + esi
    if ecx <= x:
        result = 0
        if ecx >= x:
            return result
        else:
            # should not entering here! the returning number can't be 0 any more
            # if x > 7, the program will be here
            result = func4(x=x, esi=ecx+1, edx=edx)
            return 2*result + 1
    else:
        result = func4(x=x, esi=esi, edx=ecx-1)
        return 2*result
```

Read all of the codes and comments above carefully, we know:

- we should input two numbers, the first one should not big than 14
- the second number should be zero
- The result of `func4` must be zero
- if x > 7, the recursive function returns a non-zero number, boom!
- we can easily find that if `x == 7`, the `func4` directly return 0

What's more, we can try cases from 0 to 7, and get the answers(pick one of them):

```c
0 0
1 0
3 0
7 0
```

## phase_5(doing)

Dump of assembler code for function phase_5:

```c
0x00401062 <+0>:     push   %rbx
0x00401063 <+1>:     sub    $0x20,%rsp
0x00401067 <+5>:     mov    %rdi,%rbx
// Canary usage: save %fs:0x28 to 0x18(%rsp) at the beginning
// then xor the 0x18(%rsp) at the end to see if someone attacks the program 
0x0040106a <+8>:     mov    %fs:0x28,%rax
0x00401073 <+17>:    mov    %rax,0x18(%rsp)
0x00401078 <+22>:    xor    %eax,%eax   // eax xor eax = 0
// `string_length` returns the number of characters in a string
// the string pointer is passed through %rdi (the content you input before)
// to see your string, type `x/s $rdi`
// if you are interested, `disas string_length` for more details
// you will also find how `\0` works at the end of a string
0x0040107a <+24>:    callq  0x40131b <string_length>
// so here we know we should input 6 characters
0x0040107f <+29>:    cmp    $0x6,%eax
0x00401082 <+32>:    je     0x4010d2 <phase_5+112>  //jump if %eax==6
0x00401084 <+34>:    callq  0x40143a <explode_bomb>
0x00401089 <+39>:    jmp    0x4010d2 <phase_5+112>
----------------------------------------------------------
// Note: rbx is the rdi, namely the element we input before
// movzbl: move byte to long (zero expanding)
// this means whatever we input, we only use the final byte--according to ascii
// eg: we input "iasdfg", at first we'll get 105(ascii of character `i`) 
0x0040108b <+41>:    movzbl (%rbx,%rax,1),%ecx
0x0040108f <+45>:    mov    %cl,(%rsp)      // cl is the last byte of %rcx
0x00401092 <+48>:    mov    (%rsp),%rdx
0x00401096 <+52>:    and    $0xf,%edx       // get the last 4 bits of %edx(%cl)
// x/s 0x4024b0 and get 
// "maduiersnfotvbylSo you think you can stop the bomb with ctrl-c, do you?"
// here we use te last 4 bits of %cl, adding 0x4024b0 to get the new character
// eg: character `i` gets 9 (105 = 0110 1001, last 4 bits is 9)
// then we get the 9th character of the string above, which is `f`(ascii: 102)
0x00401099 <+55>:    movzbl 0x4024b0(%rdx),%edx
0x004010a0 <+62>:    mov    %dl,0x10(%rsp,%rax,1)   // value saved in $rsp
0x004010a4 <+66>:    add    $0x1,%rax
0x004010a8 <+70>:    cmp    $0x6,%rax
0x004010ac <+74>:    jne    0x40108b <phase_5+41>   // jump if %rax!=6, loop
0x004010ae <+76>:    movb   $0x0,0x16(%rsp)
0x004010b3 <+81>:    mov    $0x40245e,%esi
0x004010b8 <+86>:    lea    0x10(%rsp),%rdi
// `x/s 0x40245e` we gets "flyers", then saved in `%esi`
// `strings_not_equal` compare two strings in `%edi` and `%esi`
// return 0 if two strings are the same 
// `disas strings_not_equal` for more details  
0x004010bd <+91>:    callq  0x401338 <strings_not_equal>
// So here the string saved in 0x10(%rsp) should be the same with "flyers"
// the index of them in "maduiersnfotvbyl..." is `9 15 14 5 6 7`
// so we should input 6 characters, the last 4 bits of which should be the values
0x004010c2 <+96>:    test   %eax,%eax
0x004010c4 <+98>:    je     0x4010d9 <phase_5+119>
0x004010c6 <+100>:   callq  0x40143a <explode_bomb>
0x004010cb <+105>:   nopl   0x0(%rax,%rax,1)
0x004010d0 <+110>:   jmp    0x4010d9 <phase_5+119>
-------------------------------------------------------
0x004010d2 <+112>:   mov    $0x0,%eax
0x004010d7 <+117>:   jmp    0x40108b <phase_5+41>
0x004010d9 <+119>:   mov    0x18(%rsp),%rax
// Canary to make sure 0x18(%rsp) is safe, since we only input 6 characters
// Here can be always safe
0x004010de <+124>:   xor    %fs:0x28,%rax
0x004010e7 <+133>:   je     0x4010ee <phase_5+140>
0x004010e9 <+135>:   callq  0x400b30 <__stack_chk_fail@plt>
0x004010ee <+140>:   add    $0x20,%rsp
0x004010f2 <+144>:   pop    %rbx
0x004010f3 <+145>:   retq
```

Read all of the codes and coments above carefully, we know:

- we should input 6 characters
- the last 4 bits of which (ascii) should be the `9 15 14 5 6 7`

So we can easily get one of the answer: `ionefg`

## Phase_6

Dump of assembler code for function phase_6:

```c
// callee saved registers, make sure do not affect the real value 
0x004010f4 <+0>:     push   %r14
0x004010f6 <+2>:     push   %r13
0x004010f8 <+4>:     push   %r12
0x004010fa <+6>:     push   %rbp
0x004010fb <+7>:     push   %rbx
0x004010fc <+8>:     sub    $0x50,%rsp
0x00401100 <+12>:    mov    %rsp,%r13
0x00401103 <+15>:    mov    %rsp,%rsi
// As we know before, after calling this, (%rsp) is the first number we input
// then following the other numbers we input, 4 bytes each
0x00401106 <+18>:    callq  0x40145c <read_six_numbers>
0x0040110b <+23>:    mov    %rsp,%r14
0x0040110e <+26>:    mov    $0x0,%r12d
0x00401114 <+32>:    mov    %r13,%rbp
// since we move address %rsp to %r13 before, here they have the same value
// key: so the first number we input should small than 6 (unsignded)
// What's more, when loop back, we'll check the second, the third...
// Then all of the numbers should be small than 6
0x00401117 <+35>:    mov    0x0(%r13),%eax
0x0040111b <+39>:    sub    $0x1,%eax
0x0040111e <+42>:    cmp    $0x5,%eax
0x00401121 <+45>:    jbe    0x401128 <phase_6+52>
0x00401123 <+47>:    callq  0x40143a <explode_bomb>
----------------------------------------------------------
// a loop here, same with `for(int r12d = 0, r12d < 6, r12d++)`
0x00401128 <+52>:    add    $0x1,%r12d
0x0040112c <+56>:    cmp    $0x6,%r12d
// when all the numbers checked: small than 6, we jump to +95
0x00401130 <+60>:    je     0x401153 <phase_6+95>
0x00401132 <+62>:    mov    %r12d,%ebx
0x00401135 <+65>:    movslq %ebx,%rax
// `movslq`: move 4 bytes to 8 bytes(signed expanding)
// here we get the value we input (%rsp + 4 * %rax)
// and it should be different with %rbp(the former value we input, 0 at first)
0x00401138 <+68>:    mov    (%rsp,%rax,4),%eax
0x0040113b <+71>:    cmp    %eax,0x0(%rbp)
0x0040113e <+74>:    jne    0x401145 <phase_6+81>   // jump if not zero
0x00401140 <+76>:    callq  0x40143a <explode_bomb>
0x00401145 <+81>:    add    $0x1,%ebx
0x00401148 <+84>:    cmp    $0x5,%ebx
0x0040114b <+87>:    jle    0x401135 <phase_6+65>   // loop back
// after this loop, all of the values are checked: not zero
// and after adding 0x4, `%r13` now starts with the next value we input
// eg: we input 1 2 3 4 5 6, and after first loop here, it is `2 3 4 5 6`
0x0040114d <+89>:    add    $0x4,%r13
0x00401151 <+93>:    jmp    0x401114 <phase_6+32>
-----------------------------------------------------
// here is a loop again: save 0x18(%rsp) to %rsi
// whatever %rsi it is, we'll loop through all of the values we input
// then make the number we input = 7 - the number we input
// eg: we input 1 2 3 4 5 6, then it will be `6 5 4 3 2 1`
0x00401153 <+95>:    lea    0x18(%rsp),%rsi
0x00401158 <+100>:   mov    %r14,%rax   // rax = the numbers we input now
0x0040115b <+103>:   mov    $0x7,%ecx
0x00401160 <+108>:   mov    %ecx,%edx
0x00401162 <+110>:   sub    (%rax),%edx // edx = 7 - the number we input
0x00401164 <+112>:   mov    %edx,(%rax)
0x00401166 <+114>:   add    $0x4,%rax   // make the pointer plus 4
0x0040116a <+118>:   cmp    %rsi,%rax
0x0040116d <+121>:   jne    0x401160 <phase_6+108>
---------------------------------------------------------
0x0040116f <+123>:   mov    $0x0,%esi
0x00401174 <+128>:   jmp    0x401197 <phase_6+163>
// %rdx is the node array, when plus 0x8, it moves to the next node
// when %ecx(7 - the number we input) matches, end the loop,
// then save the address to `0x20(%rsp,%rsi,2)` by index
// eg: %ecx = 6, it will keep going and saved the sixth node(0x603320)
// if %ecx = 1, it will just end and saved the second node(0x6032e0)
0x00401176 <+130>:   mov    0x8(%rdx),%rdx
0x0040117a <+134>:   add    $0x1,%eax
0x0040117d <+137>:   cmp    %ecx,%eax
0x0040117f <+139>:   jne    0x401176 <phase_6+130>
0x00401181 <+141>:   jmp    0x401188 <phase_6+148>
0x00401183 <+143>:   mov    $0x6032d0,%edx
/*
0x6032d0 <node1>:  0x0000014c      0x00000001      0x006032e0      0x
0x6032e0 <node2>:  0x000000a8      0x00000002      0x006032f0      0x
0x6032f0 <node3>:  0x0000039c      0x00000003      0x00603300      0x
0x603300 <node4>:  0x000002b3      0x00000004      0x00603310      0x
0x603310 <node5>:  0x000001dd      0x00000005      0x00603320      0x
0x603320 <node6>:  0x000001bb      0x00000006      0x      0x 
*/
0x00401188 <+148>:   mov    %rdx,0x20(%rsp,%rsi,2)
0x0040118d <+153>:   add    $0x4,%rsi
0x00401191 <+157>:   cmp    $0x18,%rsi
0x00401195 <+161>:   je     0x4011ab <phase_6+183>
0x00401197 <+163>:   mov    (%rsp,%rsi,1),%ecx
// get the value of number array by index
// if it is not bigger than 1 (just 1 here), jump to +143(to get first node)
// otherwise (for 2~6), jump to +130 and get the rest of nodes
0x0040119a <+166>:   cmp    $0x1,%ecx
0x0040119d <+169>:   jle    0x401183 <phase_6+143>
0x0040119f <+171>:   mov    $0x1,%eax
0x004011a4 <+176>:   mov    $0x6032d0,%edx
0x004011a9 <+181>:   jmp    0x401176 <phase_6+130>
-------------------------------------------------------
// when entering here, according to the number we input, the address of nodes
// are saved in $rsp respectively
// for example, if we input 1 2 3 4 5 6, then it will be 6 5 4 3 2 1
// which means the sixth node, the fifth ... the first node
// then (%rsp + 0x20) now is:
// 0x7fffffffe1b0: 0x00603320      0x      0x00603310      0x
// 0x7fffffffe1c0: 0x00603300      0x      0x006032f0      0x
// 0x7fffffffe1d0: 0x006032e0      0x      0x006032d0      0x
0x004011ab <+183>:   mov    0x20(%rsp),%rbx
0x004011b0 <+188>:   lea    0x28(%rsp),%rax
0x004011b5 <+193>:   lea    0x50(%rsp),%rsi
// Note: mov 0x20(%rsp) saves the value(eg, 0x603320) to %rbx
// lea 0x28(%rsp) saves the address(eg, 0x7fffffffe1b8) to %rax
0x004011ba <+198>:   mov    %rbx,%rcx
0x004011bd <+201>:   mov    (%rax),%rdx
// here it puts the address of %rdx in 0x8(%rcx)
// and according to the context, the %rdx is always the next node of %rcx
// just like `rcx.next = rdx` in C
0x004011c0 <+204>:   mov    %rdx,0x8(%rcx)
0x004011c4 <+208>:   add    $0x8,%rax
// here is a loop again, equal to `for(rax=0x20, rax!=0x50, rax+=0x8)`
0x004011c8 <+212>:   cmp    %rsi,%rax
0x004011cb <+215>:   je     0x4011d2 <phase_6+222>
0x004011cd <+217>:   mov    %rdx,%rcx
0x004011d0 <+220>:   jmp    0x4011bd <phase_6+201>
----------------------------------------------------------------------
0x004011d2 <+222>:   movq   $0x0,0x8(%rdx)
// when entering this place, the relationship of nodes is like this
// which means the list of nodes has the correct order
// if you input 1 2 3 4 5 6 and then `x/24w $rdx`
/* 
0x6032d0 <node1>: 0x0000014c  0x00000001  0x  0x
0x6032e0 <node2>: 0x000000a8  0x00000002  0x006032d0  0x
0x6032f0 <node3>: 0x0000039c  0x00000003  0x006032e0  0x
0x603300 <node4>: 0x000002b3  0x00000004  0x006032f0  0x
0x603310 <node5>: 0x000001dd  0x00000005  0x00603300  0x
0x603320 <node6>: 0x000001bb  0x00000006  0x00603310  0x
*/
// and the first node is %rbx, which saves `node6`
0x004011da <+230>:   mov    $0x5,%ebp
0x004011df <+235>:   mov    0x8(%rbx),%rax
0x004011e3 <+239>:   mov    (%rax),%eax
// here the former node should bigger than the latter one
0x004011e5 <+241>:   cmp    %eax,(%rbx)
0x004011e7 <+243>:   jge    0x4011ee <phase_6+250>
0x004011e9 <+245>:   callq  0x40143a <explode_bomb>
0x004011ee <+250>:   mov    0x8(%rbx),%rbx
0x004011f2 <+254>:   sub    $0x1,%ebp
0x004011f5 <+257>:   jne    0x4011df <phase_6+235>
// survive if you can execute here! Congratulations!
0x004011f7 <+259>:   add    $0x50,%rsp
0x004011fb <+263>:   pop    %rbx
0x004011fc <+264>:   pop    %rbp
0x004011fd <+265>:   pop    %r12
0x004011ff <+267>:   pop    %r13
0x00401201 <+269>:   pop    %r14
0x00401203 <+271>:   retq
```

Read all of the codes and comments above carefully, we know:

- we should input 6 numbers, smaller than 7, the next number should be different with the former one
- whatever we input, the numbers will be `7 - number`, eg: `1 2 3 4 5 6` -> `6 5 4 3 2 1`
- the new numbers will be used as index, to get the nodes and construct a list
- the former element in the list should bigger than the latter one

According to the node table:

```c
0x6032d0 <node1>: 0x0000014c  0x00000001  0x  0x
0x6032e0 <node2>: 0x000000a8  0x00000002  0x006032d0  0x
0x6032f0 <node3>: 0x0000039c  0x00000003  0x006032e0  0x
0x603300 <node4>: 0x000002b3  0x00000004  0x006032f0  0x
0x603310 <node5>: 0x000001dd  0x00000005  0x00603300  0x
0x603320 <node6>: 0x000001bb  0x00000006  0x00603310  0x
```

The order should be `3 4 5 6 1 2`

So we can get the answer: `4 3 2 1 6 5`

## Seems to be the end? secret_phase

Haha, **secret_phase** is waiting for you!

By `objdump -d bomb > bomb.asm`, we can get lots of codes, one thing that is:

```c
00401242 <secret_phase>:
  401242: 53                    push   %rbx
  ....
  401292: c3                    retq
```

Let's see how to enter the secret_phase

```c
004015c4 <phase_defused>:
  ...
  401630: e8 0d fc ff ff       callq  401242 <secret_phase>
```

each time when we call for `phase_defused`, it may enter the secret_phase!

Dump of assembler code for function `phase_defused`:

```c
0x004015c4 <+0>:     sub    $0x78,%rsp
0x004015c8 <+4>:     mov    %fs:0x28,%rax
0x004015d1 <+13>:    mov    %rax,0x68(%rsp)     // protect 0x68(%rsp) by Canary
0x004015d6 <+18>:    xor    %eax,%eax
0x004015d8 <+20>:    cmpl   $0x6,0x202181(%rip) # 0x603760 <num_input_strings>
// here 0x202181(%rip) gets the value of 0x603760
// this variable stores the number of strings we've input
// so after 6 phases, we may enter the secret phase
0x004015df <+27>:    jne    0x40163f <phase_defused+123>
0x004015e1 <+29>:    lea    0x10(%rsp),%r8
0x004015e6 <+34>:    lea    0xc(%rsp),%rcx
0x004015eb <+39>:    lea    0x8(%rsp),%rdx
0x004015f0 <+44>:    mov    $0x402619,%esi
0x004015f5 <+49>:    mov    $0x603870,%edi
// sscanf get the format in %esi, and get the value in %edi
// here `x/s 0x402619` we get "%d %d %s" and `x/s 0x603870` we get `7 0`
// and according to conclusion we get above
// the %rax stores the number of elements we input
// So, here we know in phase_4, we should not only input 7 0, but also another %s
0x004015fa <+54>:    callq  0x400bf0 <__isoc99_sscanf@plt>
0x004015ff <+59>:    cmp    $0x3,%eax
0x00401602 <+62>:    jne    0x401635 <phase_defused+113>
0x00401604 <+64>:    mov    $0x402622,%esi
0x00401609 <+69>:    lea    0x10(%rsp),%rdi
// `x/s 0x402622` we get "DrEvil" 0x10(%rsp) is now 0x7fffffffe1a0
// `x/s 0x7fffffffe1a0` we know this is the additional string we input
// for example, you input 7 0 asd in phase_4, and "asd" is saved in 0x10(%rsp)
// so now we know the answer: 7 0 DrEvil
0x0040160e <+74>:    callq  0x401338 <strings_not_equal>
0x00401613 <+79>:    test   %eax,%eax
0x00401615 <+81>:    jne    0x401635 <phase_defused+113>
0x00401617 <+83>:    mov    $0x4024f8,%edi
// `x/s 0x4024f8`: "Curses, you've found the secret phase!"
// function `puts` prints the string 
0x0040161c <+88>:    callq  0x400b10 <puts@plt>
0x00401621 <+93>:    mov    $0x402520,%edi
// `x/s 0x402520`: "But finding it and solving it are quite different..."
0x00401626 <+98>:    callq  0x400b10 <puts@plt>
0x0040162b <+103>:   mov    $0x0,%eax
// Congratulations! you now enter the secret_phase!
0x00401630 <+108>:   callq  0x401242 <secret_phase>
0x00401635 <+113>:   mov    $0x402558,%edi
// `x/s 0x402558` we get "Congratulations! You've defused the bomb!"
0x0040163a <+118>:   callq  0x400b10 <puts@plt>
0x0040163f <+123>:   mov    0x68(%rsp),%rax
0x00401644 <+128>:   xor    %fs:0x28,%rax
0x0040164d <+137>:   je     0x401654 <phase_defused+144>
0x0040164f <+139>:   callq  0x400b30 <__stack_chk_fail@plt>
0x00401654 <+144>:   add    $0x78,%rsp
0x00401658 <+148>:   retq
```

Dump of assembler code for function secret_phase:

```c
0x00401242 <+0>:     push   %rbx
0x00401243 <+1>:     callq  0x40149e <read_line>
// after calling this function, %rax saves the string we input
0x00401248 <+6>:     mov    $0xa,%edx
0x0040124d <+11>:    mov    $0x0,%esi
0x00401252 <+16>:    mov    %rax,%rdi
0x00401255 <+19>:    callq  0x400bd0 <strtol@plt>
// here `strtol` converts str to long, the value saved in %rax
// if the string we input is not a number, %rax will be 0, and explode then
// Note: -1 is a very large number in unsigned
0x0040125a <+24>:    mov    %rax,%rbx
0x0040125d <+27>:    lea    -0x1(%rax),%eax
// here we know the number we input should big than 1, small than 1001
0x00401260 <+30>:    cmp    $0x3e8,%eax
0x00401265 <+35>:    jbe    0x40126c <secret_phase+42>
0x00401267 <+37>:    callq  0x40143a <explode_bomb>
0x0040126c <+42>:    mov    %ebx,%esi
0x0040126e <+44>:    mov    $0x6030f0,%edi
// here %esi is the number we input, %edi is $0x6030f0 (list of node again)
// x/120 0x6030f0
/*
0x6030f0 <n1>:     0x00000024 0x00000000 0x00603110 0x00000000
0x603100 <n1+16>:  0x00603130 0x00000000 0x00000000 0x00000000
0x603110 <n21>:    0x00000008 0x00000000 0x00603190 0x00000000
0x603120 <n21+16>: 0x00603150 0x00000000 0x00000000 0x00000000
0x603130 <n22>:    0x00000032 0x00000000 0x00603170 0x00000000
0x603140 <n22+16>: 0x006031b0 0x00000000 0x00000000 0x00000000
0x603150 <n32>:    0x00000016 0x00000000 0x00603270 0x00000000
0x603160 <n32+16>: 0x00603230 0x00000000 0x00000000 0x00000000
0x603170 <n33>:    0x0000002d 0x00000000 0x006031d0 0x00000000
0x603180 <n33+16>: 0x00603290 0x00000000 0x00000000 0x00000000
0x603190 <n31>:    0x00000006 0x00000000 0x006031f0 0x00000000
0x6031a0 <n31+16>: 0x00603250 0x00000000 0x00000000 0x00000000
0x6031b0 <n34>:    0x0000006b 0x00000000 0x00603210 0x00000000
0x6031c0 <n34+16>: 0x006032b0 0x00000000 0x00000000 0x00000000
0x6031d0 <n45>:    0x00000028 0x00000000 0x00000000 0x00000000
0x6031e0 <n45+16>: 0x00000000 0x00000000 0x00000000 0x00000000
0x6031f0 <n41>:    0x00000001 0x00000000 0x00000000 0x00000000
0x603200 <n41+16>: 0x00000000 0x00000000 0x00000000 0x00000000
0x603210 <n47>:    0x00000063 0x00000000 0x00000000 0x00000000
0x603220 <n47+16>: 0x00000000 0x00000000 0x00000000 0x00000000
0x603230 <n44>:    0x00000023 0x00000000 0x00000000 0x00000000
0x603240 <n44+16>: 0x00000000 0x00000000 0x00000000 0x00000000
0x603250 <n42>:    0x00000007 0x00000000 0x00000000 0x00000000
0x603260 <n42+16>: 0x00000000 0x00000000 0x00000000 0x00000000
0x603270 <n43>:    0x00000014 0x00000000 0x00000000 0x00000000
0x603280 <n43+16>: 0x00000000 0x00000000 0x00000000 0x00000000
0x603290 <n46>:    0x0000002f 0x00000000 0x00000000 0x00000000
0x6032a0 <n46+16>: 0x00000000 0x00000000 0x00000000 0x00000000
0x6032b0 <n48>:    0x000003e9 0x00000000 0x00000000 0x00000000
0x6032c0 <n48+16>: 0x00000000 0x00000000 0x00000000 0x00000000
*/
0x00401273 <+49>:    callq  0x401204 <fun7>
// the result of func7 should be 2
0x00401278 <+54>:    cmp    $0x2,%eax
0x0040127b <+57>:    je     0x401282 <secret_phase+64>
0x0040127d <+59>:    callq  0x40143a <explode_bomb>
0x00401282 <+64>:    mov    $0x402438,%edi
// `x/s 0x402438`: "Wow! You've defused the secret stage!"
0x00401287 <+69>:    callq  0x400b10 <puts@plt>
0x0040128c <+74>:    callq  0x4015c4 <phase_defused>
0x00401291 <+79>:    pop    %rbx
0x00401292 <+80>:    retq   
```

Dump of assembler code for function fun7:

```c
   0x00401204 <+0>:     sub    $0x8,%rsp
   0x00401208 <+4>:     test   %rdi,%rdi
   0x0040120b <+7>:     je     0x401238 <fun7+52>
   0x0040120d <+9>:     mov    (%rdi),%edx
   0x0040120f <+11>:    cmp    %esi,%edx
   0x00401211 <+13>:    jle    0x401220 <fun7+28>
   0x00401213 <+15>:    mov    0x8(%rdi),%rdi
   0x00401217 <+19>:    callq  0x401204 <fun7>
   0x0040121c <+24>:    add    %eax,%eax
   0x0040121e <+26>:    jmp    0x40123d <fun7+57>
   0x00401220 <+28>:    mov    $0x0,%eax
   0x00401225 <+33>:    cmp    %esi,%edx
   0x00401227 <+35>:    je     0x40123d <fun7+57>
   0x00401229 <+37>:    mov    0x10(%rdi),%rdi
   0x0040122d <+41>:    callq  0x401204 <fun7>
   0x00401232 <+46>:    lea    0x1(%rax,%rax,1),%eax
   0x00401236 <+50>:    jmp    0x40123d <fun7+57>
   0x00401238 <+52>:    mov    $0xffffffff,%eax
   0x0040123d <+57>:    add    $0x8,%rsp
   0x00401241 <+61>:    retq
```

As we know above, it's quite hard to understand recursive funcs in assembly, so we translate it into python

```py
# edi: $0x6030f0(the memory value of it is 36) at first recursion
# x: the number we input
def func7(edi: int) -> int:
    if edi == 0:
        return 0xffffffff   # -1
    else:
        edx = MEMORY[edi]   # get the value of address edi
        if edx <= x:
            if edx == x:
                return 0
            else:
                edi = MEMORY[0x10+edi]
                return 2 * func7(edi = edi) + 1
        else:
            edi = MEMORY[0x8 + edi]
            return 2 * func7(edi = edi)
```

Read all of the codes and comments above carefully, we know:

- Type `7 0 DrEvil` we can enter the secret phase
- The result of `func7` should be 2

How can we get the 2 from `func7`? The simplest way:

1. `return 2 * func7(edi = edi)`, here the `func7` returns 1
2. `return 2 * func7(edi = edi) + 1`, here the `func7` returns 0
3. `if edx == x: return 0`

So we can use this instructions to get the rules:

- Firstly, the x should small than `edx`(MEMORY[`0x006030f0`]=36), edi now equal to MEMORY[`0x006030f0` + `0x8`] = `0x00603110`
- Secondly, the x should big than `edx`(MEMORY[`0x00603110`]=8), edi now equal to MEMORY[`0x00603110` + `0x10`] = `0x00603150`
- Finally, the x should equal to `edx`(MEMORY[`0x00603150`]=22)

So `x=22` is one of the result, you can also find other results using the guides above.

Congratulations!

## The results for copy

```c
Border relations with Canada have never been better.
1 2 4 8 16 32
6 682
7 0 DrEvil
ionefg
4 3 2 1 6 5
22
```

---
*Confused about some of the content? Feel free to report an issue [here](https://github.com/yewentao256/yewentao256.github.io/issues/new).*
