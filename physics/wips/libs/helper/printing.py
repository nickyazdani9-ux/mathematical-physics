def mprint(title, expr, break_char="="):
    from sympy import pretty_print
    print()
    print("|-- " + title + " -- |" + "\n")
    print(break_char * 60)
    pretty_print(expr)
    print(break_char * 60)
