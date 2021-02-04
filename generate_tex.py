colors = ["red", "orange", "green", "blue", "violet"]

files_dir = "data"
output_dir = "figures"

plots = [
    {
        "title": "\\texttt{Hatespeech} \\textbf{BB}",
        "output": "hatespeech_bb.tex",
        "xmin": 0,
        "xmax": 6,
        "xtick": [0, 2, 4, 6],
        "series": [
            {
                "file": "hatespeech_brendel",
                "name": "original",
            },
            *[
                {
                    "file": f"hatespeech_stable_{e}_brendel",
                    "name": f"stable {e}"
                } for e in ("0.90", "0.89", "0.88", "0.86")
            ]
        ]
    },
    {
        "title": "\\texttt{Hatespeech} \\textbf{JSMA}",
        "output": "hatespeech_jsma.tex",
        "xmin": 0,
        "xmax": 6,
        "xtick": [0, 2, 4, 6],
        "series": [
            {
                "file": "hatespeech_custom_jsma",
                "name": "original",
            },
            *[
                {
                    "file": f"hatespeech_stable_{e}_custom_jsma",
                    "name": f"stable {e}"
                } for e in ("0.90", "0.89", "0.88", "0.86")
            ]
        ]
    },
    {
        "title": "\\texttt{PDFRate} \\textbf{BB}",
        "output": "pdfrate_bb.tex",
        "xmin": 0,
        "xmax": 20,
        "xtick": [0, 4, 8, 12, 16, 20],
        "series": [
            {
                "file": "pdfrate_brendel",
                "name": "original",
            },
            *[
                {
                    "file": f"pdfrate_stable_{e}_brendel",
                    "name": f"stable {e}"
                } for e in ("0.99", "0.985", "0.98", "0.96")
            ]
        ]
    },
    {
        "title": "\\texttt{PDFRate} \\textbf{JSMA}",
        "output": "pdfrate_jsma.tex",
        "xmin": 0,
        "xmax": 20,
        "xtick": [0, 4, 8, 12, 16, 20],
        "series": [
            {
                "file": "pdfrate_custom_jsma",
                "name": "original",
            },
            *[
                {
                    "file": f"pdfrate_stable_{e}_custom_jsma",
                    "name": f"stable {e}"
                } for e in ("0.99", "0.985", "0.98", "0.96")
            ]
        ]
    },
    {
        "title": "\\texttt{Hidost} \\textbf{JSMA}",
        "output": "hidost_jsma.tex",
        "xmin": 0,
        "xmax": 80,
        "xtick": [0, 20, 40, 60, 80],
        "series": [
            {
                "file": "hidost_custom_jsma",
                "name": "original",
            },
            *[
                {
                    "file": f"hidost_stable_{e}_custom_jsma",
                    "name": f"stable {e}"
                } for e in ("0.996", "0.993", "0.99", "0.98")
            ]
        ]
    },
    {
        "title": "\\texttt{Hidost} \\textbf{BB}",
        "output": "hidost_bb.tex",
        "xmin": 0,
        "xmax": 80,
        "xtick": [0, 20, 40, 60, 80],
        "series": [
            {
                "file": "hidost_brendel",
                "name": "original",
            },
            *[
                {
                    "file": f"hidost_stable_{e}_brendel",
                    "name": f"stable {e}"
                } for e in ("0.996", "0.993", "0.99", "0.98")
            ]
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