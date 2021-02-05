colors = ["black", "red", "blue", "green", "orange"]

files_dir = "data"
output_dir = "figures"

no_acc = True

f = "no_acc" if no_acc else ""
t = "GMB" if no_acc else "GMBC"

plots = [
    {
        "title": f"\\texttt{{Hatespeech}} \\textbf{{BB}} {t}",
        "output": f"hatespeech_bb{f}.tex",
        "xmin": 0,
        "xmax": 6,
        "xtick": [0, 1, 2, 3, 4, 5, 6],
        "series": [
            {
                "file": "hatespeech_brendel",
                "name": "original",
            },
            *[
                {
                    "file": f"hatespeech_stable_{f}{e}_brendel",
                    "name": f"$\\beta$ = {e}"
                } for e in ("0.90", "0.89", "0.88")
            ]
        ]
    },
    {
        "title": f"\\texttt{{Hatespeech}} \\textbf{{JSMA}} {t}",
        "output": f"hatespeech_jsma{f}.tex",
        "xmin": 0,
        "xmax": 6,
        "xtick": [0, 1, 2, 3, 4, 5, 6],
        "series": [
            {
                "file": "hatespeech_custom_jsma",
                "name": "original",
            },
            *[
                {
                    "file": f"hatespeech_stable_{f}{e}_custom_jsma",
                    "name": f"$\\beta$ = {e}"
                } for e in ("0.90", "0.89", "0.88")
            ]
        ]
    },
    {
        "title": f"\\texttt{{PDFRate}} \\textbf{{BB}} {t}",
        "output": f"pdfrate_bb{f}.tex",
        "xmin": 0,
        "xmax": 28,
        "xtick": [0, 4, 8, 12, 16, 20, 24, 28],
        "series": [
            {
                "file": "pdfrate_brendel",
                "name": "original",
            },
            {
                "file": f"pdfrate_stable_{f}0.99_brendel",
                "name": "$\\beta$ = 0.990"
            },
            {
                "file": f"pdfrate_stable_{f}0.985_brendel",
                "name": "$\\beta$ = 0.985"
            },
            {
                "file": f"pdfrate_stable_{f}0.98_brendel",
                "name": "$\\beta$ = 0.980"
            },
        ]
    },
    {
        "title": f"\\texttt{{PDFRate}} \\textbf{{JSMA}} {t}",
        "output": f"pdfrate_jsma{f}.tex",
        "xmin": 0,
        "xmax": 28,
        "xtick": [0, 4, 8, 12, 16, 20, 24, 28],
        "series": [
            {
                "file": "pdfrate_custom_jsma",
                "name": "original",
            },
            {
                "file": f"pdfrate_stable_{f}0.99_custom_jsma",
                "name": "$\\beta$ = 0.990"
            },
            {
                "file": f"pdfrate_stable_{f}0.985_custom_jsma",
                "name": "$\\beta$ = 0.985"
            },
            {
                "file": f"pdfrate_stable_{f}0.98_custom_jsma",
                "name": "$\\beta$ = 0.980"
            },
        ]
    },
    {
        "title": f"\\texttt{{Hidost}} \\textbf{{JSMA}} {t}",
        "output": f"hidost_jsma{f}.tex",
        "xmin": 0,
        "xmax": 60,
        "xtick": [0, 12, 24, 36, 48, 60],
        "series": [
            {
                "file": "hidost_custom_jsma",
                "name": "original",
            },
            {
                "file": f"hidost_stable_{f}0.996_custom_jsma",
                "name": "$\\beta$ = 0.995"
            },
            {
                "file": f"hidost_stable_{f}0.99_custom_jsma",
                "name": "$\\beta$ = 0.990"
            }
        ]
    },
    {
        "title": f"\\texttt{{Hidost}} \\textbf{{BB}} {t}",
        "output": f"hidost_bb{f}.tex",
        "xmin": 0,
        "xmax": 60,
        "xtick": [0, 12, 24, 36, 48, 60],
        "series": [
            {
                "file": "hidost_brendel",
                "name": "original",
            },
            {
                "file": f"hidost_stable_{f}0.996_brendel",
                "name": "$\\beta$ = 0.995"
            },
            {
                "file": f"hidost_stable_{f}0.99_brendel",
                "name": "$\\beta$ = 0.990"
            }
        ]
    }
]

import os

for plot in plots:

    title = plot["title"]
    output = plot["output"]
    xmin = plot["xmin"]
    xmax = plot["xmax"]
    xtick = plot["xtick"]
    series = plot["series"]

    xtick_text = ",".join([str(i) for i in xtick])

    series_text_array = [
        f"""\\addplot[color={colors[index]},style=semithick]
table {{{os.path.join(files_dir, serie["file"] + ".txt")}}};
\\addlegendentry{{{serie["name"]}}}"""
        for index, serie in enumerate(series)
    ]

    series_text = "\n".join(series_text_array)

    tex_code=f"""\\documentclass[tikz]{{standalone}}
%%% aidan's packages %%%
% for graphing
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.16}}

\\usepackage{{tcolorbox}} %%% Must be after pgfplots due to option clash.

\\begin{{document}}
\\begin{{tikzpicture}}
\\begin{{axis}}[
xmin=0, xmax={xmax}, xtick={{{xtick_text}}},
ymin=0, ymax=1,
ylabel = \\large Accuracy,
xlabel = \\large $\\epsilon$,
height = 6cm,
title=\\large{title}
]
{series_text}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{document}}
"""

    with open(os.path.join(output_dir, output), "w") as out_file:
        out_file.write(tex_code)