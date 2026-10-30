"""Write manuscript.tex: main.tex with every \\input inlined and no graphicspath."""
import re, pathlib
here = pathlib.Path(__file__).parent
def expand(text):
    def rep(m):
        return expand((here / (m.group(1) + '.tex')).read_text()).rstrip('\n')
    return re.sub(r'^\\input\{([^}]+)\}', rep, text, flags=re.M)
s = expand((here / 'main.tex').read_text())
s = s.replace('\\graphicspath{{figures/}}\n', '')
(here / 'manuscript.tex').write_text(s)
