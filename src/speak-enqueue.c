/*
 * speak-enqueue: Fast fire-and-forget TTS enqueue client.
 * Sends a length-prefixed JSON message to speak-daemon's Unix socket.
 *
 * Usage: speak-enqueue [-v voice] [-s speed] [-c caller] TEXT...
 *    or: echo TEXT | speak-enqueue [-v voice] [-s speed] [-c caller]
 *
 * Compile: cc -O2 -o bin/speak-enqueue src/speak-enqueue.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <getopt.h>
#include <errno.h>

#define MAX_TEXT 65536
#define MAX_JSON (MAX_TEXT + 512)

static char *get_sock_path(void) {
    static char buf[256];
    const char *p = getenv("SPEAK_SOCK");
    if (p) return (char *)p;
    const char *user = getenv("USER");
    if (!user) user = "unknown";
    snprintf(buf, sizeof(buf), "/tmp/speak-%s.sock", user);
    return buf;
}

static int json_escape(const char *src, char *dst, int dstlen) {
    int j = 0;
    for (int i = 0; src[i] && j < dstlen - 2; i++) {
        switch (src[i]) {
            case '"':  dst[j++] = '\\'; dst[j++] = '"'; break;
            case '\\': dst[j++] = '\\'; dst[j++] = '\\'; break;
            case '\n': dst[j++] = '\\'; dst[j++] = 'n'; break;
            case '\r': dst[j++] = '\\'; dst[j++] = 'r'; break;
            case '\t': dst[j++] = '\\'; dst[j++] = 't'; break;
            default:   dst[j++] = src[i]; break;
        }
    }
    dst[j] = '\0';
    return j;
}

int main(int argc, char *argv[]) {
    const char *voice = "af_heart", *speed = "1.26", *caller = NULL;
    int opt;
    while ((opt = getopt(argc, argv, "v:s:c:")) != -1) {
        switch (opt) {
            case 'v': voice = optarg; break;
            case 's': speed = optarg; break;
            case 'c': caller = optarg; break;
            default:
                fprintf(stderr, "Usage: speak-enqueue [-v voice] [-s speed] [-c caller] TEXT...\n");
                return 1;
        }
    }

    char text[MAX_TEXT];
    text[0] = '\0';
    if (optind < argc) {
        int off = 0;
        for (int i = optind; i < argc && off < MAX_TEXT - 2; i++) {
            if (i > optind) text[off++] = ' ';
            off += snprintf(text + off, MAX_TEXT - off, "%s", argv[i]);
        }
    } else if (!isatty(STDIN_FILENO)) {
        int n = fread(text, 1, MAX_TEXT - 1, stdin);
        text[n] = '\0';
        /* trim trailing whitespace */
        while (n > 0 && (text[n-1] == '\n' || text[n-1] == '\r' || text[n-1] == ' '))
            text[--n] = '\0';
    } else {
        fprintf(stderr, "speak-enqueue: no text\n");
        return 1;
    }
    if (!text[0]) {
        fprintf(stderr, "speak-enqueue: empty text\n");
        return 1;
    }

    char escaped[MAX_TEXT * 2];
    json_escape(text, escaped, sizeof(escaped));

    char json[MAX_JSON];
    int json_len;
    if (caller && caller[0])
        json_len = snprintf(json, sizeof(json),
            "{\"enqueue\":true,\"text\":\"%s\",\"voice\":\"%s\",\"speed\":%s,\"caller\":\"%s\"}",
            escaped, voice, speed, caller);
    else
        json_len = snprintf(json, sizeof(json),
            "{\"enqueue\":true,\"text\":\"%s\",\"voice\":\"%s\",\"speed\":%s}",
            escaped, voice, speed);

    /* connect to daemon */
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        perror("speak-enqueue: socket");
        return 1;
    }
    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, get_sock_path(), sizeof(addr.sun_path) - 1);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        if (errno == ENOENT || errno == ECONNREFUSED)
            fprintf(stderr, "speak-enqueue: daemon not running (start with: speak --daemon)\n");
        else
            perror("speak-enqueue: connect");
        close(fd);
        return 1;
    }

    /* send length-prefixed JSON */
    uint32_t net_len = htonl(json_len);
    write(fd, &net_len, 4);
    write(fd, json, json_len);

    /* read response */
    uint32_t resp_len;
    char resp[4096];
    if (read(fd, &resp_len, 4) == 4) {
        resp_len = ntohl(resp_len);
        if (resp_len > 0 && resp_len < sizeof(resp)) {
            int total = 0;
            while (total < (int)resp_len) {
                int n = read(fd, resp + total, resp_len - total);
                if (n <= 0) break;
                total += n;
            }
            resp[total] = '\0';
            /* print position to stderr like Python client does */
            char *pos = strstr(resp, "\"position\"");
            if (pos) {
                char *colon = strchr(pos, ':');
                if (colon) {
                    int p = atoi(colon + 1);
                    if (p > 0) fprintf(stderr, "queued (position %d)\n", p);
                }
            }
            if (!strstr(resp, "\"ok\"") || strstr(resp, "false")) {
                fprintf(stderr, "speak-enqueue: %s\n", resp);
                close(fd);
                return 1;
            }
        }
        /* consume zero terminator */
        read(fd, &resp_len, 4);
    }

    close(fd);
    return 0;
}
