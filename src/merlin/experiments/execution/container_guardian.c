/* Trusted command owner inside an already isolated PID namespace.
 * No target semantics, compiler seed, service credential or admission authority.
 * All child output is encoded; only this process writes the result protocol.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define LIMIT 65536
static unsigned char out[LIMIT], err[LIMIT];
static size_t used[2];
static volatile sig_atomic_t interrupted;

static void stop(int signal_number) { interrupted = signal_number; }
static double seconds(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value)) _exit(125);
  return value.tv_sec + value.tv_nsec / 1000000000.0;
}
static void hex(const unsigned char *bytes, size_t size) {
  for (size_t index = 0; index < size; ++index) printf("%02x", bytes[index]);
}
static int capabilities_dropped(void) {
  FILE *source = fopen("/proc/self/status", "r");
  char line[256]; int found = 0;
  if (!source) return 0;
  while (fgets(line, sizeof(line), source)) {
    unsigned long long value;
    if (sscanf(line, "CapEff: %llx", &value) == 1) found = value == 0;
  }
  fclose(source); return found;
}
int main(int argc, char **argv) {
  char *end;
  if (argc < 4 || strcmp(argv[2], "--") || getpid() != 1 || !capabilities_dropped()) return 125;
  errno = 0; double budget = strtod(argv[1], &end);
  if (errno || *end || !(budget > 0 && budget <= 3600)) return 125;
  if (prctl(PR_SET_DUMPABLE, 0) || prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) return 125;
  struct sigaction action = {0};
  action.sa_handler = stop; sigemptyset(&action.sa_mask);
  if (sigaction(SIGTERM, &action, NULL) || sigaction(SIGINT, &action, NULL)) return 125;
  int streams[2][2];
  if (pipe2(streams[0], O_CLOEXEC) || pipe2(streams[1], O_CLOEXEC)) return 125;
  pid_t child = fork();
  if (child < 0) return 125;
  if (!child) {
    if (setsid() < 0 || dup2(streams[0][1], STDOUT_FILENO) < 0 || dup2(streams[1][1], STDERR_FILENO) < 0) _exit(125);
    int null_input = open("/dev/null", O_RDONLY);
    if (null_input < 0 || dup2(null_input, STDIN_FILENO) < 0) _exit(125);
    /* The engine starts this private owner without extra descriptors. Close all
     * inherited pipe/control descriptors before arbitrary compiler execution. */
    if (close_range(3, ~0U, 0)) _exit(125);
    if (clearenv() || setenv("HOME", "/tmp", 1) || setenv("PATH", "/usr/bin:/bin", 1)
        || setenv("PYTHONPATH", "/component-inputs/compiler", 1)
        || setenv("PYTHONDONTWRITEBYTECODE", "1", 1) || setenv("PYTHONNOUSERSITE", "1", 1)) _exit(125);
    execvp(argv[3], argv + 3); _exit(127);
  }
  for (int index = 0; index < 2; ++index) {
    close(streams[index][1]);
    if (fcntl(streams[index][0], F_SETFL, O_NONBLOCK) < 0) return 125;
  }
  struct pollfd descriptors[2] = {{streams[0][0], POLLIN, 0}, {streams[1][0], POLLIN, 0}};
  double deadline = seconds() + budget;
  int child_done = 0, child_status = 0, exceeded = 0, timed_out = 0;
  while (!child_done) {
    pid_t observed = waitpid(child, &child_status, WNOHANG);
    if (observed == child) { child_done = 1; break; }
    if (observed < 0 && errno != EINTR) return 125;
    if (interrupted || seconds() >= deadline) { timed_out = 1; break; }
    if (poll(descriptors, 2, 10) < 0 && errno != EINTR) return 125;
    for (int index = 0; index < 2; ++index) {
      unsigned char buffer[4096]; ssize_t size;
      while ((size = read(streams[index][0], buffer, sizeof(buffer))) > 0) {
        if ((size_t)size > LIMIT - used[index]) { exceeded = 1; break; }
        memcpy((index ? err : out) + used[index], buffer, size); used[index] += size;
      }
      if (exceeded) break;
    }
    if (exceeded) break;
  }
  /* PID 1 sees only this command namespace. It cannot signal host processes;
   * capability removal and no-new-privileges prevent changing child ownership.
   * Kernel init adoption covers double forks and new sessions. */
  if (kill(-1, SIGKILL) < 0 && errno != ESRCH) return 125;
  int reaped = 0, status;
  for (;;) {
    pid_t observed = waitpid(-1, &status, 0);
    if (observed > 0) {
      ++reaped;
      if (observed == child) { child_done = 1; child_status = status; }
      continue;
    }
    if (observed < 0 && errno == EINTR) continue;
    if (observed < 0 && errno == ECHILD) break;
    return 125;
  }
  for (int index = 0; index < 2; ++index) {
    unsigned char buffer[4096]; ssize_t size;
    while ((size = read(streams[index][0], buffer, sizeof(buffer))) > 0) {
      if ((size_t)size > LIMIT - used[index]) { exceeded = 1; break; }
      memcpy((index ? err : out) + used[index], buffer, size); used[index] += size;
    }
    close(streams[index][0]);
  }
  if (!child_done) return 125;
  int code = WIFEXITED(child_status) ? WEXITSTATUS(child_status) : 128 + WTERMSIG(child_status);
  printf("{\"schema\":\"merlin.container_command.v1\",\"pid\":1,\"returncode\":%d,", code);
  printf("\"timed_out\":%s,\"output_exceeded\":%s,\"reaped_after_command\":%d,", timed_out ? "true" : "false", exceeded ? "true" : "false", reaped);
  printf("\"cleanup_wait_echild\":true,\"stdout_hex\":\""); hex(out, used[0]);
  printf("\",\"stderr_hex\":\""); hex(err, used[1]); puts("\"}");
  return 0;
}
