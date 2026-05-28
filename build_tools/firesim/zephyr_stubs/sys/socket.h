/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Minimal stub for <sys/socket.h> on bare-metal newlib RISC-V targets.
 *
 * iree_bar's runtime/src/iree/async/socket.c unconditionally includes
 * <sys/socket.h> in its POSIX branch (no IREE_PLATFORM_GENERIC bypass),
 * which breaks any newlib bare-metal build (firesim profile too -- the
 * pre-existing firesim build artifact predates the iree async/ landing
 * and is stale).
 *
 * This stub provides just enough of the BSD sockets surface for socket.c
 * and its siblings to compile. The resulting object files contain
 * unresolved references to socket(), bind(), listen(), accept(),
 * sendmsg(), recvmsg(), connect(), shutdown(), setsockopt(),
 * getsockopt(), poll(), epoll_*. Those symbols are *never* resolved at
 * the Zephyr-application link step: the local-sync HAL never
 * instantiates an iree_async_proactor_t, so the linker GC drops the
 * iree_async_async object slices that depend on them
 * (-ffunction-sections + --gc-sections in the Zephyr link).
 *
 * If you ever wire iree_async into a Zephyr app, replace this with the
 * Zephyr POSIX <sys/socket.h> (CONFIG_NET_SOCKETS=y, CONFIG_POSIX_API=y).
 */

#ifndef MERLIN_ZEPHYR_STUB_SYS_SOCKET_H_
#define MERLIN_ZEPHYR_STUB_SYS_SOCKET_H_

#include <stddef.h>
#include <stdint.h>

/* socklen_t is sometimes already provided by newlib via <sys/types.h>; if
 * so, this typedef is harmless because we use the same width. */
#if !defined(__socklen_t_defined) && !defined(_SOCKLEN_T_DECLARED)
typedef uint32_t socklen_t;
#define __socklen_t_defined 1
#define _SOCKLEN_T_DECLARED 1
#endif

typedef uint16_t sa_family_t;

struct sockaddr {
	sa_family_t sa_family;
	char sa_data[14];
};

struct sockaddr_storage {
	sa_family_t ss_family;
	char __padding[126];
};

struct iovec {
	void *iov_base;
	size_t iov_len;
};

struct msghdr {
	void *msg_name;
	socklen_t msg_namelen;
	struct iovec *msg_iov;
	size_t msg_iovlen;
	void *msg_control;
	size_t msg_controllen;
	int msg_flags;
};

struct cmsghdr {
	size_t cmsg_len;
	int cmsg_level;
	int cmsg_type;
};

#define AF_UNSPEC 0
#define AF_UNIX 1
#define AF_INET 2
#define AF_INET6 10

#define SOCK_STREAM 1
#define SOCK_DGRAM 2
#define SOCK_RAW 3
#define SOCK_SEQPACKET 5
#define SOCK_NONBLOCK 0x800
#define SOCK_CLOEXEC 0x80000

#define SOL_SOCKET 1
#define SO_REUSEADDR 2
#define SO_TYPE 3
#define SO_ERROR 4
#define SO_BROADCAST 6
#define SO_SNDBUF 7
#define SO_RCVBUF 8
#define SO_KEEPALIVE 9
#define SO_LINGER 13
#define SO_RCVTIMEO 20
#define SO_SNDTIMEO 21

#define MSG_OOB 0x01
#define MSG_PEEK 0x02
#define MSG_DONTROUTE 0x04
#define MSG_TRUNC 0x20
#define MSG_DONTWAIT 0x40
#define MSG_EOR 0x80
#define MSG_WAITALL 0x100
#define MSG_NOSIGNAL 0x4000

#define SHUT_RD 0
#define SHUT_WR 1
#define SHUT_RDWR 2

#define SOMAXCONN 128
#define FIONBIO 0x5421
#define IPPROTO_TCP 6
#define IPPROTO_UDP 17
#define TCP_NODELAY 1

#ifdef __cplusplus
extern "C" {
#endif

int socket(int domain, int type, int protocol);
int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
int listen(int sockfd, int backlog);
int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen);
int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
int shutdown(int sockfd, int how);
int setsockopt(
	int sockfd, int level, int optname, const void *optval, socklen_t optlen);
int getsockopt(
	int sockfd, int level, int optname, void *optval, socklen_t *optlen);
int getsockname(int sockfd, struct sockaddr *addr, socklen_t *addrlen);
int getpeername(int sockfd, struct sockaddr *addr, socklen_t *addrlen);

long send(int sockfd, const void *buf, size_t len, int flags);
long recv(int sockfd, void *buf, size_t len, int flags);
long sendto(int sockfd, const void *buf, size_t len, int flags,
	const struct sockaddr *dest_addr, socklen_t addrlen);
long recvfrom(int sockfd, void *buf, size_t len, int flags,
	struct sockaddr *src_addr, socklen_t *addrlen);
long sendmsg(int sockfd, const struct msghdr *msg, int flags);
long recvmsg(int sockfd, struct msghdr *msg, int flags);

#ifdef __cplusplus
}
#endif

#endif /* MERLIN_ZEPHYR_STUB_SYS_SOCKET_H_ */
