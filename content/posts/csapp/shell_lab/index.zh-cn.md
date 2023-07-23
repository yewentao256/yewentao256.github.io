---
title: "ShellLab"
date: 2023-06-23T10:41:24+08:00
categories: ["csapp"]
summary: "在这个lab中，我们将通过开发一个简洁但完整的unix shell程序来对进程控制、信号机制等概念有进一步的认知。源码：[https://github.com/yewentao256/CSAPP_15213/tree/main/shelllab]"
---

## Summary

在这个lab中，我们将通过开发一个简洁但完整的unix shell程序来对进程控制、信号机制等概念有进一步的认知。源码地址：[https://github.com/yewentao256/CSAPP_15213/tree/main/shelllab]

## 等待被翻译

非常抱歉，看起来这篇博文还没有被翻译成中文，请等待一段时间

## How to launch(Using docker)

Source from [Yansongsongsong](https://github.com/Yansongsongsong/CSAPP-Experiments)

Firstly using a docker:

`docker run -d -p 9912:22 --name shelllab yansongsongsong/csapp:shelllab`

Then using vscode plugin **remote ssh**

`ssh root@127.0.0.1 -p 9912`

password: `THEPASSWORDYOUCREATED`

## How to validate

`./sdriver.pl -t trace01.txt -s ./tsh -a "-p"` or `make test01`

You can compare the result with `tshref`

`./sdriver.pl -t trace01.txt -s ./tshref -a "-p"` or `make rtest01`

## Trace 01

To shutdown the program when meeting `EOF` (`ctrl-D` in linux)

It's developed already.

## Trace 02

To shutdown the program using `quit` command. Only need to update `eval` function here.

```c
void eval(char *cmdline) 
{
    char * argv[MAXARGS];       /* args list */
    parseline(cmdline, argv);       /* parse command line to argv */

    if (argv[0] == NULL) return;    /* ignore empty command */

    if (!strcmp(argv[0], "quit")) {
        exit(0);   /* exit the shell */
    }
}
```

## Trace 03

Here we need to call `/bin/echo` binary and output `"tsh> quit"`, then `quit`

This can be realized by using `execve`

```c
void eval(char *cmdline) {
  char *argv[MAXARGS];      /* args list */
  parseline(cmdline, argv); /* parse command line to argv */
  pid_t pid;

  if (argv[0] == NULL) return; /* ignore empty command */

  if (!builtin_cmd(argv)) {
    /* not a builtin command */
    if ((pid = fork()) == 0) {
      /* sub process execute here */
      if (execve(argv[0], argv, environ) == -1) {
        /* not found */
        printf("%s: command not found \n", argv[0]);
        exit(0);
      }
    }
  }
}
```

Note that we've move the `quit` function to `builtin_cmd`

```c
int builtin_cmd(char **argv) {
  if (!strcmp(argv[0], "quit")) {
    exit(0); /* exit the shell */
  }
  return 0; /* not a builtin command */
}
```

## Trace 04

Here we need to support background job: `./myspin 1 &`

So we should use `addjob()`, `deletejob()` to manage jobs. And parse `&` by using `parseline()` to distinguish foreground job and background job.

What's more, we need to realize `sigchld_handler` to reap zombie subprocess.

```c
void eval(char *cmdline) {
  char *argv[MAXARGS];
  int bg;              /* whether runs in background */
  pid_t pid;

  bg = parseline(cmdline, argv); /* parse command line to argv */
  if (argv[0] == NULL) return;   /* ignore empty command */

  if (!builtin_cmd(argv)) {
    if ((pid = fork()) == 0) {
      if (execve(argv[0], argv, environ) == -1) {
        printf("%s: command not found \n", argv[0]);
        exit(0);
      }
    } else {
      /* parrent executes here */
      addjob(jobs, pid, bg == 1 ? BG : FG, cmdline);

      if (!bg) {
        waitfg(pid); /* wait foreground job */
      } else {
        printf("[%d] (%d) %s", pid2jid(pid), pid, cmdline);
      }
    }
  }
}

void waitfg(pid_t pid) {
  while (pid == fgpid(jobs)) {
    sleep(0);
  }
}

void sigchld_handler(int sig) {
  pid_t pid;
  /* pid_t waitpid(pid_t pid, int *status, int options);
     pid == -1 means waiting for arbitrary sub process
     WHOHANG means that if no process exits, return 0 */
  while ((pid = waitpid(-1, NULL, WNOHANG)) > 0) {
    deletejob(jobs, pid);
  }
}
```

Looks all good, right? **However, `fork()` may execute so fast that it exits before parent calls `addjob`, which can cause trouble.** The parent may think that the subprocess is still running and keep waiting for it, while the subprocess has already finished.

So we should block `SIGCHLD` signal before parent calls `addjob` to avoid this problem.

```c
void eval(char *cmdline) {
  // ...
  sigset_t sig_mask_child, oldset; /* signal set, unsigned long actually */

  if (!builtin_cmd(argv)) {
    sigemptyset(&sig_mask_child);        /* set sigset all zero*/
    sigaddset(&sig_mask_child, SIGCHLD); /* add SIGCHLD to sig set*/

    /* block signal SIGCHLD */
    sigprocmask(SIG_BLOCK, &sig_mask_child, &oldset);

    if ((pid = fork()) == 0) {
      /* recover from blocking signal for child, we should do this because 
         children inherit the blocked vectors of their parents */
      sigprocmask(SIG_SETMASK, &oldset, NULL);
      // execve()...
    } else {
      /* parrent executes here */
      addjob(jobs, pid, bg == 1 ? BG : FG, cmdline);

      /* recover from blocking signal for parent */
      sigprocmask(SIG_SETMASK, &oldset, NULL);

      // ...
    }
  }
}
```

## Trace 05

Trace 5 is quite simple here, the only thing we need to do is to implment the `jobs` command.

```c
int builtin_cmd(char **argv) {
  // ...
  if (!strcmp(argv[0], "jobs")) {
    listjobs(jobs); /* print jobs */
    exit(0);
  }
  return 0; /* not a builtin command */
}
```

## Trace 06

We need to realize `sigint_handler` here.

The easiest way is like this:

```c
void sigint_handler(int sig) {
  /* find foreground pid */
  pid_t pid;
  if ((pid = fgpid(jobs)) > 0) {
    printf("Job [%d] (%d) terminated by signal 2\n", pid2jid(pid), pid);
    kill(pid, sig);
    deletejob(jobs, pid);
  }
}
```

Note: we only kill the foreground process using `kill(pid, sig)` here, but not for the whole **group** by using `-pid` in `kill` function. This is an issue and we'll solve it later.

## Trace 07

We should forward `SIGINT` only to foreground job.

As we use `kill(pid, sig)` above instead of using `kill(-pid, sig)`, we can still pass this trace.

This is an issue since the subprocesses of foreground job are not cared, we'll solve it later.

## Trace 08

We should forward SIGTSTP only to foreground job here.

If we realize this like:

```c
void sigtstp_handler(int sig) {
  pid_t pid;
  if ((pid = fgpid(jobs)) > 0) {
    printf("Job [%d] (%d) stopped by signal 20\n", pid2jid(pid), pid);
    kill(pid, sig);
    struct job_t * job = getjobpid(jobs, pid);
    job->state = ST;
  }
}
```

we'll find that the process is hanging, traping into a deadlock. Let's figure out what's happening

```bash
ps ajf
 PPID   PID  PGID   SID TTY      TPGID STAT   UID   TIME COMMAND
  131  8983  8983  8983 pts/11    9748 Ss       0   0:00 /bin/zsh -i
 8983  9748  9748  8983 pts/11    9748 R+       0   0:00  \_ ps ajf
  131   679   679   679 pts/9     9666 Ss       0   0:01 /bin/zsh -i
  679  9666  9666   679 pts/9     9666 S+       0   0:00  \_ make test08
 9666  9667  9666   679 pts/9     9666 S+       0   0:00      \_ /usr/bin/perl ./sdriver.pl -t trace08.
 9667  9668  9666   679 pts/9     9666 R+       0   0:02          \_ ./tsh -p
 9668  9672  9666   679 pts/9     9666 T+       0   0:00              \_ ./myspin 5
```

We can find that only `pid=9672` is suspended, while other processes are still running! And since we are `waitfg(pid /* 9672 */);`, here we are trapped.

The solution of this problem is by using `kill(-pid, sig)` to send signal to the process **group**.

However, we should forward SIGTSTP **only** to foreground job. So we need to make a change of our code, furthermore, using `setpgid()` to put subprocess into a new process group.

Here is the code:

```c
void eval(char *cmdline) {
  // ...
  if (!builtin_cmd(argv)) {
    // ...
    if ((pid = fork()) == 0) {
      /* set subprocess into a new process group */
      if (setpgid(0, 0) == -1) {
        perror("setpgid");
        exit(EXIT_FAILURE);
      }
      /* sub process executes here */
    }
    // ...
  }
}
```

Also, update `kill` in `sigint_handler` to fix the issue mentioned in trace 06 and trace 07

```c
void sigint_handler(int sig) {
  // ...
  kill(-pid, sig);
}
```

## Trace 09

Here we need to process bg builtin command

Firstly adding it in builtin command

```c
int builtin_cmd(char **argv) {
  // ...
  if (!strcmp(argv[0], "bg")) {
    do_bgfg(argv); /* do bg command */
    return 1;
  }
  return 0; /* not a builtin command */
}
```

Then realize the `do_bgfg` function

```c
void do_bgfg(char **argv) {
  int jid;
  pid_t pid;
  struct job_t *job;

  if (argv[1][0] == '%') {
    /* bg/fg %1 means job id = 1 */
    jid = atoi(argv[1] + 1);
    job = getjobjid(jobs, jid);
    pid = job->pid;
  } else {
    /* bg/fg 1 means process id = 1 */
    pid = atoi(argv[1]);
    job = getjobpid(jobs, pid);
    jid = job->jid;
  }

  if (!strcmp(argv[0], "bg")) {
    printf("[%d] [%d] %s", jid, pid, job->cmdline);
    if (kill(-pid, SIGCONT) == -1) {
      perror("Sending SIGCONT to bg job failed");
      exit(EXIT_FAILURE);
    }
    job->state = BG;
  }

}
```

## Trace 10

Here we need to process fg builtin command.

Similar to Trace 09, adding `fg` in `builtin_cmd`

```c++
 if (!strcmp(argv[0], "bg") || !strcmp(argv[0], "fg")) {
    do_bgfg(argv); /* do bg command */
    return 1;
  }
```

Then fulfill the `do_bgfg` function

```c
  if (!strcmp(argv[0], "bg")) {
    // ...
  } else {
    if (job->state == ST) {
      if (kill(-pid, SIGCONT) == -1) {
        perror("Sending SIGCONT to fg job failed");
        exit(EXIT_FAILURE);
      }
    }
    job->state = FG;
    waitfg(pid);
  }
```

## Trace 11

Here we need to forward SIGINT to every process in foreground process group

Since we create process group for every process (See Trace 08), and sending `kill(-pid, ...)` to every process in group. Nothing needs to do here.

## Trace 12

Forward SIGTSTP to every process in foreground process group

Same as above, nothing needs to do here.

## Trace 13

Restart every stopped process in process group

Same as above, nothing needs to do here.

## Trace 14

Here we should handle possible error.

We should consider case when:

- `argv[1] == NULL`
- `argv[1]` is not a number
- `argv[1]` doesn't matches current jobid/process id.

Here is the code:

```c
void do_bgfg(char **argv) {
  int jid;
  pid_t pid;
  struct job_t *job;

  if (argv[1] == NULL) {
    printf("%s command requires PID or %%jobid argument\n", argv[0]);
    return;
  }

  if (argv[1][0] == '%') {
    /* %1 means job id = 1 */
    if (strspn(argv[1] + 1, "0123456789") != strlen(argv[1]) - 1) {
      /* check all the characters are numbers */
      printf("argument must be a PID or %%jobid\n");
      return;
    }
    jid = atoi(argv[1] + 1);
    job = getjobjid(jobs, jid);
    if (job == NULL) {
      printf("(%d): No such job\n", jid);
      return;
    }
    pid = job->pid;
  } else {
    /* process id */
    if (strspn(argv[1], "0123456789") != strlen(argv[1])) {
      /* check all the characters are numbers */
      printf("argument must be a PID or %%jobid\n");
      return;
    }
    pid = atoi(argv[1]);
    job = getjobpid(jobs, pid);
    if (job == NULL) {
      printf("(%d): No such process\n", pid);
      return;
    }
    jid = job->jid;
  }

  // ...
}
```

## Trace 15

Putting it all together.

We directly pass this trace, too.

## Trace16

Here the shell are supposed to handle SIGTSTP and SIGINT signals that come from other processes instead of the terminal.

We try to execute test16 at first, but then we find that we are trapped in `./mystop`

```bash
make test16
./sdriver.pl -t trace16.txt -s ./tsh -a "-p"
#
# trace16.txt - Tests whether the shell can handle SIGTSTP and SIGINT
#     signals that come from other processes instead of the terminal.
#
tsh> ./mystop 2
# blocking....
```

`ps ajf` to figure out why:

```bash
 PPID   PID  PGID   SID TTY      TPGID STAT   UID   TIME COMMAND
10361 10463 10463 10463 pts/4    16940 Ss       0   0:00 /bin/zsh -i
10463 16940 16940 10463 pts/4    16940 S+       0   0:00  \_ make test16
16940 16941 16940 10463 pts/4    16940 S+       0   0:00      \_ /usr/bin/perl ./sdriver.pl
16941 16942 16940 10463 pts/4    16940 S+       0   0:00          \_ ./tsh -p
16942 16944 16944 10463 pts/4    16940 T        0   0:00              \_ ./mystop 2
```

Here the `./mystop 2` process has been stopped by it self. Howerver, our `sigtstp_handler` doesn't capture the signal, and the job is still in `FG` state.

To fix this issue, we should update the codes in `sigchld_handler` to receive the signal sent by subprocess. We can realize this by adding `WUNTRACED` in `waitpid()`

```c
void sigchld_handler(int sig) {
  pid_t pid;
  int status;
  struct job_t *job;
  /* `pid_t waitpid(pid_t pid, int *status, int options)`
     pid == -1 means waiting for arbitrary sub process
     WHOHANG means that if no process exits, return 0
     WUNTRACED means if subprocess is suspended, return its pid */
  while ((pid = waitpid(-1, &status, WNOHANG | WUNTRACED)) > 0) {
    job = getjobpid(jobs, pid);
    if (WIFEXITED(status)) {
      deletejob(jobs, pid);
    } else if (WIFSTOPPED(status)) {
      printf("Job [%d] (%d) stopped by signal %d\n", job->jid, pid,
             WSTOPSIG(status));
      job->state = ST;
    }
  }
}
```

Fix the issue above!

Then we find that we can't handle the `./myint 2` command. Same as above, adding logic to `sigchld_handler` to handle this:

```c
while ((pid = waitpid(-1, &status, WNOHANG | WUNTRACED)) > 0) {
    job = getjobpid(jobs, pid);
    if (WIFEXITED(status)) {
      // ...
    } else if (WIFSIGNALED(status)) {
      printf("Job [%d] (%d) terminated by signal %d\n", job->jid, pid, WTERMSIG(status));
      deletejob(jobs, pid);
    } else if (WIFSTOPPED(status)) {
      // ...
    }
  }
```

Since we've handled `deletejob` in `sigchld_handler()`, we should remove it from current `sigint_handler()`. And the same idea, remove `job->state = ST;` in `sigtstp_handler()`

Update like this:

```c
void sigint_handler(int sig) {
  /* find foreground pid */
  pid_t pid;
  if ((pid = fgpid(jobs)) > 0) {
    kill(-pid, sig);
  }
}

void sigtstp_handler(int sig) {
  pid_t pid;
  if ((pid = fgpid(jobs)) > 0) {
    kill(-pid, sig);
  }
}
```

Finish this trace!

## Review

Note: all the handlers logic are managed in parent process(`tsh shell`), so the calling chain of handlers may like this:

User types control-c -> parent handles in `sigint_handler`, forwarding the signal to subprocess -> subprocess killed, sending signal to parent -> parent handles it in `sigchld_handler`.

To make sure every trace can be passed in our newest code, we execute all of them again.

And all of the traces passed. Take trace 15 as an example:

```bash
➜  shlab-handout make rtest15
./sdriver.pl -t trace15.txt -s ./tshref -a "-p"
#
# trace15.txt - Putting it all together
#
tsh> ./bogus
./bogus: Command not found
tsh> ./myspin 10
Job [1] (23928) terminated by signal 2
tsh> ./myspin 3 &
[1] (23940) ./myspin 3 &
tsh> ./myspin 4 &
[2] (23943) ./myspin 4 &
tsh> jobs
[1] (23940) Running ./myspin 3 &
[2] (23943) Running ./myspin 4 &
tsh> fg %1
Job [1] (23940) stopped by signal 20
tsh> jobs
[1] (23940) Stopped ./myspin 3 &
[2] (23943) Running ./myspin 4 &
tsh> bg %3
%3: No such job
tsh> bg %1
[1] (23940) ./myspin 3 &
tsh> jobs
[1] (23940) Running ./myspin 3 &
[2] (23943) Running ./myspin 4 &
tsh> fg %1
tsh> quit
➜  shlab-handout make test15 
./sdriver.pl -t trace15.txt -s ./tsh -a "-p"
#
# trace15.txt - Putting it all together
#
tsh> ./bogus
./bogus: Command not found 
tsh> ./myspin 10
Job [1] (23992) terminated by signal 2
tsh> ./myspin 3 &
[1] (24015) ./myspin 3 &
tsh> ./myspin 4 &
[2] (24017) ./myspin 4 &
tsh> jobs
[1] (24015) Running ./myspin 3 &
[2] (24017) Running ./myspin 4 &
tsh> fg %1
Job [1] (24015) stopped by signal 20
tsh> jobs
[1] (24015) Stopped ./myspin 3 &
[2] (24017) Running ./myspin 4 &
tsh> bg %3
(3): No such job
tsh> bg %1
[1] [24015] ./myspin 3 &
tsh> jobs
[1] (24015) Running ./myspin 3 &
[2] (24017) Running ./myspin 4 &
tsh> fg %1
tsh> quit
```

Note that we didn't block signals in `sigchld_handler()`, and we can still pass the traces. However, we want to make our code better(and stronger), let's add it to `sigchld_handler()`.

If we don't add it, in some cornor case(quite unlikely), when parent is dealing with one CHLD signal and receives INT or another CHLD signal, and both of them requires `deletejob()`, we will be in trouble.

Here's the code:

```c
void sigchld_handler(int sig) {
  pid_t pid;
  int status;
  struct job_t *job;

  /* signal blocker */
  sigset_t mask_all, prev_all;
  sigfillset(&mask_all);

  /* `pid_t waitpid(pid_t pid, int *status, int options)`
     pid == -1 means waiting for arbitrary sub process
     WHOHANG means that if no process exits, return 0
     WUNTRACED means if subprocess is suspended, return its pid */
  while ((pid = waitpid(-1, &status, WNOHANG | WUNTRACED)) > 0) {
    if (sigprocmask(SIG_BLOCK, &mask_all, &prev_all) < 0) {
      perror("sigprocmask error");
      exit(1);
    }
    job = getjobpid(jobs, pid);
    if (WIFEXITED(status)) {
      deletejob(jobs, pid);
    } else if (WIFSIGNALED(status)) {
      printf("Job [%d] (%d) terminated by signal %d\n", job->jid, pid,
             WTERMSIG(status));
      deletejob(jobs, pid);
    } else if (WIFSTOPPED(status)) {
      printf("Job [%d] (%d) stopped by signal %d\n", job->jid, pid,
             WSTOPSIG(status));
      job->state = ST;
    }
    if (sigprocmask(SIG_SETMASK, &prev_all, NULL) < 0) {
      perror("sigprocmask error");
      exit(1);
    }
  }
}
```

## Appendix

A complete `tsh.c`:

```c
/*
 * tsh - A tiny shell program with job control
 *
 * Author: Peter
 */
#include <ctype.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

/* Misc manifest constants */
#define MAXLINE 1024   /* max line size */
#define MAXARGS 128    /* max args on a command line */
#define MAXJOBS 16     /* max jobs at any point in time */
#define MAXJID 1 << 16 /* max job ID */

/* Job states */
#define UNDEF 0 /* undefined */
#define FG 1    /* running in foreground */
#define BG 2    /* running in background */
#define ST 3    /* stopped */

/*
 * Jobs states: FG (foreground), BG (background), ST (stopped)
 * Job state transitions and enabling actions:
 *     FG -> ST  : ctrl-z
 *     ST -> FG  : fg command
 *     ST -> BG  : bg command
 *     BG -> FG  : fg command
 * At most 1 job can be in the FG state.
 */

/* Global variables */
extern char **environ;   /* defined in libc */
char prompt[] = "tsh> "; /* command line prompt (DO NOT CHANGE) */
int verbose = 0;         /* if true, print additional output */
int nextjid = 1;         /* next job ID to allocate */
char sbuf[MAXLINE];      /* for composing sprintf messages */

struct job_t {           /* The job struct */
  pid_t pid;             /* job PID */
  int jid;               /* job ID [1, 2, ...] */
  int state;             /* UNDEF, BG, FG, or ST */
  char cmdline[MAXLINE]; /* command line */
};
struct job_t jobs[MAXJOBS]; /* The job list */
/* End global variables */

/* Function prototypes */

/* Here are the functions that you will implement */
void eval(char *cmdline);
int builtin_cmd(char **argv);
void do_bgfg(char **argv);
void waitfg(pid_t pid);

void sigchld_handler(int sig);
void sigtstp_handler(int sig);
void sigint_handler(int sig);

/* Here are helper routines that we've provided for you */
int parseline(const char *cmdline, char **argv);
void sigquit_handler(int sig);

void clearjob(struct job_t *job);
void initjobs(struct job_t *jobs);
int maxjid(struct job_t *jobs);
int addjob(struct job_t *jobs, pid_t pid, int state, char *cmdline);
int deletejob(struct job_t *jobs, pid_t pid);
pid_t fgpid(struct job_t *jobs);
struct job_t *getjobpid(struct job_t *jobs, pid_t pid);
struct job_t *getjobjid(struct job_t *jobs, int jid);
int pid2jid(pid_t pid);
void listjobs(struct job_t *jobs);

void usage(void);
void unix_error(char *msg);
void app_error(char *msg);
typedef void handler_t(int);
handler_t *Signal(int signum, handler_t *handler);

/*
 * main - The shell's main routine
 */
int main(int argc, char **argv) {
  char c;
  char cmdline[MAXLINE];
  int emit_prompt = 1; /* emit prompt (default) */

  /* Redirect stderr to stdout (so that driver will get all output
   * on the pipe connected to stdout) */
  dup2(1, 2);

  /* Parse the command line */
  while ((c = getopt(argc, argv, "hvp")) != EOF) {
    switch (c) {
      case 'h': /* print help message */
        usage();
        break;
      case 'v': /* emit additional diagnostic info */
        verbose = 1;
        break;
      case 'p':          /* don't print a prompt */
        emit_prompt = 0; /* handy for automatic testing */
        break;
      default:
        usage();
    }
  }

  /* Install the signal handlers */

  /* These are the ones you will need to implement */
  Signal(SIGINT, sigint_handler);   /* ctrl-c */
  Signal(SIGTSTP, sigtstp_handler); /* ctrl-z */
  Signal(SIGCHLD, sigchld_handler); /* Terminated or stopped child */

  /* This one provides a clean way to kill the shell */
  Signal(SIGQUIT, sigquit_handler);

  /* Initialize the job list */
  initjobs(jobs);

  /* Execute the shell's read/eval loop */
  while (1) {
    /* Read command line */
    if (emit_prompt) {
      printf("%s", prompt);
      fflush(stdout);
    }
    if ((fgets(cmdline, MAXLINE, stdin) == NULL) && ferror(stdin))
      app_error("fgets error");
    if (feof(stdin)) { /* End of file (ctrl-d) */
      fflush(stdout);
      exit(0);
    }

    /* Evaluate the command line */
    eval(cmdline);
    fflush(stdout);
    fflush(stdout);
  }

  exit(0); /* control never reaches here */
}

/*
 * eval - Evaluate the command line that the user has just typed in
 *
 * If the user has requested a built-in command (quit, jobs, bg or fg)
 * then execute it immediately. Otherwise, fork a child process and
 * run the job in the context of the child. If the job is running in
 * the foreground, wait for it to terminate and then return.  Note:
 * each child process must have a unique process group ID so that our
 * background children don't receive SIGINT (SIGTSTP) from the kernel
 * when we type ctrl-c (ctrl-z) at the keyboard.
 */
void eval(char *cmdline) {
  char *argv[MAXARGS];             /* args list */
  int bg;                          /* whether runs in background */
  pid_t pid;                       /* subprocess pid */
  sigset_t sig_mask_child, oldset; /* signal set, unsigned long actually */

  bg = parseline(cmdline, argv); /* parse command line to argv */
  if (argv[0] == NULL) return;   /* ignore empty command */

  if (!builtin_cmd(argv)) {
    sigemptyset(&sig_mask_child);        /* set sigset all zero*/
    sigaddset(&sig_mask_child, SIGCHLD); /* add SIGCHLD to sig set*/

    /* block signal SIGCHLD */
    sigprocmask(SIG_BLOCK, &sig_mask_child, &oldset);
    /* not a builtin command */
    if ((pid = fork()) == 0) {
      /* set subprocess into a new process group */
      if (setpgid(0, 0) == -1) {
        perror("setpgid");
        exit(EXIT_FAILURE);
      }
      /* sub process executes here */
      /* recover from blocking signal for child, we should do this because
         children inherit the blocked vectors of their parents */
      sigprocmask(SIG_SETMASK, &oldset, NULL);
      if (execve(argv[0], argv, environ) == -1) {
        printf("%s: Command not found \n", argv[0]);
        exit(0);
      }
    } else {
      /* parrent executes here */
      addjob(jobs, pid, bg == 1 ? BG : FG, cmdline);

      /* recover from blocking signal for parent */
      sigprocmask(SIG_SETMASK, &oldset, NULL);

      if (!bg) {
        waitfg(pid); /* wait foreground job */
      } else {
        printf("[%d] (%d) %s", pid2jid(pid), pid, cmdline);
      }
    }
  }
}

/*
 * parseline - Parse the command line and build the argv array.
 *
 * Characters enclosed in single quotes are treated as a single
 * argument.  Return true if the user has requested a BG job, false if
 * the user has requested a FG job.
 */
int parseline(const char *cmdline, char **argv) {
  static char array[MAXLINE]; /* holds local copy of command line */
  char *buf = array;          /* ptr that traverses command line */
  char *delim;                /* points to first space delimiter */
  int argc;                   /* number of args */
  int bg;                     /* background job? */

  strcpy(buf, cmdline);
  buf[strlen(buf) - 1] = ' ';   /* replace trailing '\n' with space */
  while (*buf && (*buf == ' ')) /* ignore leading spaces */
    buf++;

  /* Build the argv list */
  argc = 0;
  if (*buf == '\'') {
    buf++;
    delim = strchr(buf, '\'');
  } else {
    delim = strchr(buf, ' ');
  }

  while (delim) {
    argv[argc++] = buf;
    *delim = '\0';
    buf = delim + 1;
    while (*buf && (*buf == ' ')) /* ignore spaces */
      buf++;

    if (*buf == '\'') {
      buf++;
      delim = strchr(buf, '\'');
    } else {
      delim = strchr(buf, ' ');
    }
  }
  argv[argc] = NULL;

  if (argc == 0) /* ignore blank line */
    return 1;

  /* should the job run in the background? */
  if ((bg = (*argv[argc - 1] == '&')) != 0) {
    argv[--argc] = NULL;
  }
  return bg;
}

/*
 * builtin_cmd - If the user has typed a built-in command then execute
 *    it immediately.
 */
int builtin_cmd(char **argv) {
  if (!strcmp(argv[0], "quit")) {
    exit(0); /* exit the shell */
  }
  if (!strcmp(argv[0], "jobs")) {
    listjobs(jobs); /* print jobs */
    return 1;
  }
  if (!strcmp(argv[0], "bg") || !strcmp(argv[0], "fg")) {
    do_bgfg(argv); /* do bg/fg command */
    return 1;
  }
  return 0; /* not a builtin command */
}

/*
 * do_bgfg - Execute the builtin bg and fg commands
 */
void do_bgfg(char **argv) {
  int jid;
  pid_t pid;
  struct job_t *job;

  if (argv[1] == NULL) {
    printf("%s command requires PID or %%jobid argument\n", argv[0]);
    return;
  }

  if (argv[1][0] == '%') {
    /* %1 means job id = 1 */
    if (strspn(argv[1] + 1, "0123456789") != strlen(argv[1]) - 1) {
      /* check all the characters are numbers */
      printf("argument must be a PID or %%jobid\n");
      return;
    }
    jid = atoi(argv[1] + 1);
    job = getjobjid(jobs, jid);
    if (job == NULL) {
      printf("(%d): No such job\n", jid);
      return;
    }
    pid = job->pid;
  } else {
    /* process id */
    if (strspn(argv[1], "0123456789") != strlen(argv[1])) {
      /* check all the characters are numbers */
      printf("argument must be a PID or %%jobid\n");
      return;
    }
    pid = atoi(argv[1]);
    job = getjobpid(jobs, pid);
    if (job == NULL) {
      printf("(%d): No such process\n", pid);
      return;
    }
    jid = job->jid;
  }

  if (!strcmp(argv[0], "bg")) {
    printf("[%d] [%d] %s", jid, pid, job->cmdline);
    if (kill(-pid, SIGCONT) == -1) {
      perror("Sending SIGCONT to bg job failed");
      exit(EXIT_FAILURE);
    }
    job->state = BG;
  } else {
    if (job->state == ST) {
      if (kill(-pid, SIGCONT) == -1) {
        perror("Sending SIGCONT to fg job failed");
        exit(EXIT_FAILURE);
      }
    }
    job->state = FG;
    waitfg(pid);
  }
}

/*
 * waitfg - Block until process pid is no longer the foreground process
 */
void waitfg(pid_t pid) {
  while (pid == fgpid(jobs)) {
    sleep(0);
  }
}

/*****************
 * Signal handlers
 *****************/

/*
 * sigchld_handler - The kernel sends a SIGCHLD to the shell whenever
 *     a child job terminates (becomes a zombie), or stops because it
 *     received a SIGSTOP or SIGTSTP signal. The handler reaps all
 *     available zombie children, but doesn't wait for any other
 *     currently running children to terminate.
 */
void sigchld_handler(int sig) {
  pid_t pid;
  int status;
  struct job_t *job;

  /* signal blocker */
  sigset_t mask_all, prev_all;
  sigfillset(&mask_all);

  /* `pid_t waitpid(pid_t pid, int *status, int options)`
     pid == -1 means waiting for arbitrary sub process
     WHOHANG means that if no process exits, return 0
     WUNTRACED means if subprocess is suspended, return its pid */
  while ((pid = waitpid(-1, &status, WNOHANG | WUNTRACED)) > 0) {
    if (sigprocmask(SIG_BLOCK, &mask_all, &prev_all) < 0) {
      perror("sigprocmask error");
      exit(1);
    }
    job = getjobpid(jobs, pid);
    if (WIFEXITED(status)) {
      deletejob(jobs, pid);
    } else if (WIFSIGNALED(status)) {
      printf("Job [%d] (%d) terminated by signal %d\n", job->jid, pid,
             WTERMSIG(status));
      deletejob(jobs, pid);
    } else if (WIFSTOPPED(status)) {
      printf("Job [%d] (%d) stopped by signal %d\n", job->jid, pid,
             WSTOPSIG(status));
      job->state = ST;
    }
    if (sigprocmask(SIG_SETMASK, &prev_all, NULL) < 0) {
      perror("sigprocmask error");
      exit(1);
    }
  }
}

/*
 * sigint_handler - The kernel sends a SIGINT to the shell whenver the
 *    user types ctrl-c at the keyboard.  Catch it and send it along
 *    to the foreground job.
 */
void sigint_handler(int sig) {
  /* find foreground pid */
  pid_t pid;
  if ((pid = fgpid(jobs)) > 0) {
    kill(-pid, sig);
  }
}

/*
 * sigtstp_handler - The kernel sends a SIGTSTP to the shell whenever
 *     the user types ctrl-z at the keyboard. Catch it and suspend the
 *     foreground job by sending it a SIGTSTP.
 */
void sigtstp_handler(int sig) {
  pid_t pid;
  if ((pid = fgpid(jobs)) > 0) {
    kill(-pid, sig);
  }
}

/*********************
 * End signal handlers
 *********************/

/***********************************************
 * Helper routines that manipulate the job list
 **********************************************/

/* clearjob - Clear the entries in a job struct */
void clearjob(struct job_t *job) {
  job->pid = 0;
  job->jid = 0;
  job->state = UNDEF;
  job->cmdline[0] = '\0';
}

/* initjobs - Initialize the job list */
void initjobs(struct job_t *jobs) {
  int i;

  for (i = 0; i < MAXJOBS; i++) clearjob(&jobs[i]);
}

/* maxjid - Returns largest allocated job ID */
int maxjid(struct job_t *jobs) {
  int i, max = 0;

  for (i = 0; i < MAXJOBS; i++)
    if (jobs[i].jid > max) max = jobs[i].jid;
  return max;
}

/* addjob - Add a job to the job list */
int addjob(struct job_t *jobs, pid_t pid, int state, char *cmdline) {
  int i;

  if (pid < 1) return 0;

  for (i = 0; i < MAXJOBS; i++) {
    if (jobs[i].pid == 0) {
      jobs[i].pid = pid;
      jobs[i].state = state;
      jobs[i].jid = nextjid++;
      if (nextjid > MAXJOBS) nextjid = 1;
      strcpy(jobs[i].cmdline, cmdline);
      if (verbose) {
        printf("Added job [%d] %d %s\n", jobs[i].jid, jobs[i].pid,
               jobs[i].cmdline);
      }
      return 1;
    }
  }
  printf("Tried to create too many jobs\n");
  return 0;
}

/* deletejob - Delete a job whose PID=pid from the job list */
int deletejob(struct job_t *jobs, pid_t pid) {
  int i;

  if (pid < 1) return 0;

  for (i = 0; i < MAXJOBS; i++) {
    if (jobs[i].pid == pid) {
      clearjob(&jobs[i]);
      nextjid = maxjid(jobs) + 1;
      return 1;
    }
  }
  return 0;
}

/* fgpid - Return PID of current foreground job, 0 if no such job */
pid_t fgpid(struct job_t *jobs) {
  int i;

  for (i = 0; i < MAXJOBS; i++)
    if (jobs[i].state == FG) return jobs[i].pid;
  return 0;
}

/* getjobpid  - Find a job (by PID) on the job list */
struct job_t *getjobpid(struct job_t *jobs, pid_t pid) {
  int i;

  if (pid < 1) return NULL;
  for (i = 0; i < MAXJOBS; i++)
    if (jobs[i].pid == pid) return &jobs[i];
  return NULL;
}

/* getjobjid  - Find a job (by JID) on the job list */
struct job_t *getjobjid(struct job_t *jobs, int jid) {
  int i;

  if (jid < 1) return NULL;
  for (i = 0; i < MAXJOBS; i++)
    if (jobs[i].jid == jid) return &jobs[i];
  return NULL;
}

/* pid2jid - Map process ID to job ID */
int pid2jid(pid_t pid) {
  int i;

  if (pid < 1) return 0;
  for (i = 0; i < MAXJOBS; i++)
    if (jobs[i].pid == pid) {
      return jobs[i].jid;
    }
  return 0;
}

/* listjobs - Print the job list */
void listjobs(struct job_t *jobs) {
  int i;

  for (i = 0; i < MAXJOBS; i++) {
    if (jobs[i].pid != 0) {
      printf("[%d] (%d) ", jobs[i].jid, jobs[i].pid);
      switch (jobs[i].state) {
        case BG:
          printf("Running ");
          break;
        case FG:
          printf("Foreground ");
          break;
        case ST:
          printf("Stopped ");
          break;
        default:
          printf("listjobs: Internal error: job[%d].state=%d ", i,
                 jobs[i].state);
      }
      printf("%s", jobs[i].cmdline);
    }
  }
}
/******************************
 * end job list helper routines
 ******************************/

/***********************
 * Other helper routines
 ***********************/

/*
 * usage - print a help message
 */
void usage(void) {
  printf("Usage: shell [-hvp]\n");
  printf("   -h   print this message\n");
  printf("   -v   print additional diagnostic information\n");
  printf("   -p   do not emit a command prompt\n");
  exit(1);
}

/*
 * unix_error - unix-style error routine
 */
void unix_error(char *msg) {
  fprintf(stdout, "%s: %s\n", msg, strerror(errno));
  exit(1);
}

/*
 * app_error - application-style error routine
 */
void app_error(char *msg) {
  fprintf(stdout, "%s\n", msg);
  exit(1);
}

/*
 * Signal - wrapper for the sigaction function
 */
handler_t *Signal(int signum, handler_t *handler) {
  struct sigaction action, old_action;

  action.sa_handler = handler;
  sigemptyset(&action.sa_mask); /* block sigs of type being handled */
  action.sa_flags = SA_RESTART; /* restart syscalls if possible */

  if (sigaction(signum, &action, &old_action) < 0) unix_error("Signal error");
  return (old_action.sa_handler);
}

/*
 * sigquit_handler - The driver program can gracefully terminate the
 *    child shell by sending it a SIGQUIT signal.
 */
void sigquit_handler(int sig) {
  printf("Terminating after receipt of SIGQUIT signal\n");
  exit(1);
}

```

---
*Confused about some of the content? Feel free to report an issue [here](https://github.com/yewentao256/yewentao256.github.io/issues/new).*
