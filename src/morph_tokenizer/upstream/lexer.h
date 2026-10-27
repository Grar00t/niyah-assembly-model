#ifndef NIYAH_LEXER_H
#define NIYAH_LEXER_H

typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long u64;
typedef long i64;

typedef enum {
    T_EOF, T_ERROR, T_IDENT, T_INT, T_STRING,
    K_IF, K_ELSE, K_WHILE, K_RETURN, K_VAR, K_CONST, K_FUNC, K_STRUCT,
    T_LPAREN, T_RPAREN, T_LBRACE, T_RBRACE, T_LBRACK, T_RBRACK,
    T_COMMA, T_SEMI, T_COLON, T_DOT, T_ARROW,
    T_PLUS, T_MINUS, T_STAR, T_SLASH, T_PERCENT,
    T_AMP, T_PIPE, T_CARET, T_TILDE, T_BANG,
    T_ASSIGN, T_EQ, T_NE, T_LT, T_LE, T_GT, T_GE,
    T_SHL, T_SHR, T_ANDAND, T_OROR,
    T_COUNT
} TokKind;

typedef struct {
    TokKind kind;
    u32 line, col;          /* 1-based; col counts code points */
    const u8 *start;        /* raw bytes in source buffer */
    u32 len;
    i64 value;              /* T_INT value; T_ERROR: message id */
} Token;

typedef struct {
    const u8 *cur, *end;
    u32 line, col;
    u8 last_cr;            /* CRLF counts as one logical newline */
} Lexer;

void lex_init(Lexer *L, const u8 *src, u64 len);
Token lex_next(Lexer *L);
const char *tok_name(TokKind k);
const char *lex_errmsg(i64 id);

#endif
