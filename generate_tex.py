colors = ["black", "red", "blue", "green", "orange"]

files_dir = "data"
output_dir = "figures"

no_acc = False

f = "no_acc" if no_acc else ""
t = "GMB" if no_acc else "GMBC"
do_at = False

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

at_plots = [
    {
        "title": f"\\texttt{{PDFRate}}(AT 1-epoch) \\textbf{{BB}} GMB",
        "output": f"pdfrate_at1_bb.tex",
        "xmin": 0,
        "xmax": 24,
        "xtick": [0, 4, 8, 12, 16, 20, 24],
        "series": [
            {
                "file": "pdfrate_at1_brendel",
                "name": "AT"
            },
            {
                "file": "pdfrate_at1_stable_brendel",
                "name": "stable"
            }
        ]

    },
    {
        "title": f"\\texttt{{PDFRate}}(AT 1-epoch) \\textbf{{JSMA}} GMB",
        "output": f"pdfrate_at1_jsma.tex",
        "xmin": 0,
        "xmax": 24,
        "xtick": [0, 4, 8, 12, 16, 20, 24],
        "series": [
            {
                "file": "pdfrate_at1_custom_jsma",
                "name": "AT"
            },
            {
                "file": "pdfrate_at1_stable_custom_jsma",
                "name": "stable"
            }
        ]
    },
    {
        "title": f"\\texttt{{PDFRate}}(AT 5-epoch) \\textbf{{BB}} GMB",
        "output": f"pdfrate_at5_bb.tex",
        "xmin": 0,
        "xmax": 24,
        "xtick": [0, 4, 8, 12, 16, 20, 24],
        "series": [
            {
                "file": "pdfrate_at5_brendel",
                "name": "AT"
            },
            {
                "file": "pdfrate_at5_stable_brendel",
                "name": "stable"
            }
        ]

    },
    {
        "title": f"\\texttt{{PDFRate}}(AT 5-epoch) \\textbf{{JSMA}} GMB",
        "output": f"pdfrate_at5_jsma.tex",
        "xmin": 0,
        "xmax": 24,
        "xtick": [0, 4, 8, 12, 16, 20, 24],
        "series": [
            {
                "file": "pdfrate_at5_custom_jsma",
                "name": "AT"
            },
            {
                "file": "pdfrate_at5_stable_custom_jsma",
                "name": "stable"
            }
        ]
    },
    {
        "title": f"\\texttt{{PDFRate}}(AT 10-epoch) \\textbf{{BB}} GMB",
        "output": f"pdfrate_at10_bb.tex",
        "xmin": 0,
        "xmax": 24,
        "xtick": [0, 4, 8, 12, 16, 20, 24],
        "series": [
            {
                "file": "pdfrate_at10_brendel",
                "name": "AT"
            },
            {
                "file": "pdfrate_at10_stable_brendel",
                "name": "stable"
            }
        ]

    },
    {
        "title": f"\\texttt{{PDFRate}}(AT 10-epoch) \\textbf{{JSMA}} GMB",
        "output": f"pdfrate_at5_jsma.tex",
        "xmin": 0,
        "xmax": 24,
        "xtick": [0, 4, 8, 12, 16, 20, 24],
        "series": [
            {
                "file": "pdfrate_at10_custom_jsma",
                "name": "AT"
            },
            {
                "file": "pdfrate_at10_stable_custom_jsma",
                "name": "stable"
            }
        ]
    },
    {
        "title": f"\\texttt{{Hidost}}(AT 1-epoch) \\textbf{{BB}} GMB",
        "output": f"hidost_at1_bb.tex",
        "xmin": 0,
        "xmax": 60,
        "xtick": [0, 12, 24, 36, 48, 60],
        "series": [
            {
                "file": "hidost_at1_brendel",
                "name": "AT"
            },
            {
                "file": "hidost_at1_stable_brendel",
                "name": "stable"
            }
        ]
    },
    {
        "title": f"\\texttt{{Hidost}}(AT 4-epoch) \\textbf{{BB}} GMB",
        "output": f"hidost_at4_bb.tex",
        "xmin": 0,
        "xmax": 60,
        "xtick": [0, 12, 24, 36, 48, 60],
        "series": [
            {
                "file": "hidost_at4_brendel",
                "name": "AT"
            },
            {
                "file": "hidost_at4_stable_brendel",
                "name": "stable"
            }
        ]
    },
    {
        "title": f"\\texttt{{Hidost}}(AT 1-epoch) \\textbf{{JSMA}} GMB",
        "output": f"hidost_at1_jsma.tex",
        "xmin": 0,
        "xmax": 60,
        "xtick": [0, 12, 24, 36, 48, 60],
        "series": [
            {
                "file": "hidost_at1_custom_jsma",
                "name": "AT"
            },
            {
                "file": "hidost_at1_stable_custom_jsma",
                "name": "stable"
            }
        ]
    },
    {
        "title": f"\\texttt{{Hidost}}(AT 4-epoch) \\textbf{{JSMA}} GMB",
        "output": f"hidost_at4_jsma.tex",
        "series": [
            {
                "file": "hidost_at4_custom_jsma",
                "name": "AT"
            },
            {
                "file": "hidost_at4_stable_custom_jsma",
                "name": "stable"
            }
        ]
    },
]

if do_at:
    plots = at_plots

import os

for plot in plots:

    title = plot["title"]
    output = plot["output"]

    if "hidost" in output:
        xmin = 0
        xmax = 80
        xtick = [0, 20, 40, 60, 80]
    elif "pdfrate" in output:
        xmin = 0
        xmax = 28
        xtick = [0, 4, 8, 12, 16, 20, 24, 28]
    else:
        assert("hatespeech" in output)
        xmin = 0
        xmax = 6
        xtick = [0, 2, 4, 6]

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