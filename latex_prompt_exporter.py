template = """
\\begin{{tcolorbox}}[title={},
    colback=red!5!white,
    colframe=red!75!black,
    colbacktitle=yellow!50!red,
    coltitle=red!25!black,
    fonttitle=\\bfseries,
    subtitle style={{boxrule=0.4pt,
    colback=yellow!50!red!25!white,
    colupper=red!75!gray}},
    parbox=false]
    \\tcbsubtitle{{Context \\texttt{{\char"0023 example input}}}}
    \\begin{{lstlisting}}[breaklines, basicstyle=\\small,  breakatwhitespace=true, breakindent=0\\dimen0]
{}
    \\end{{lstlisting}}
    \\tcbsubtitle{{Prompt \\texttt{{\char"0023 query ChatGPT}}}}
    \\begin{{lstlisting}}[breaklines, basicstyle=\\small,  breakatwhitespace=true, breakindent=0\\dimen0]
{}
    \\end{{lstlisting}}
    \\tcbsubtitle{{Expected answer}}
    \\begin{{lstlisting}}[breaklines, basicstyle=\\small,  breakatwhitespace=true, breakindent=0\\dimen0]
{}
    \\end{{lstlisting}}
\\end{{tcolorbox}}
"""


def trunc(string):
    if len(string) > 150:
        return string[:150] + "..."
    else:
        return string


def export_latex(task_name, context, prompt, label, filename):
    text_file = open(filename, "w")
    text_file.write(template.format(task_name, trunc(context), prompt.strip(), label))
    text_file.close()
