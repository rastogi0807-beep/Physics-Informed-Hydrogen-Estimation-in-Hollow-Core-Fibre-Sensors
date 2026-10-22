"""Small tested LaTeX-subset to native editable Office Math (OMML) authoring.

Only commands used in this manuscript are accepted. Unsupported syntax raises
an exception rather than leaking source markup into the Word document.
"""
import re
from copy import deepcopy
from lxml import etree
NS='http://schemas.openxmlformats.org/officeDocument/2006/math'
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def el(name,*children,**attrs):
    q=etree.Element('{'+NS+'}'+name)
    for k,v in attrs.items():q.set('{'+NS+'}'+k,str(v))
    for c in children:
        if isinstance(c,list):
            for x in c:q.append(x)
        else:q.append(c)
    return q
def run(text,plain=False):
    r=el('r')
    if plain:r.append(el('rPr',el('sty',val='p')))
    pr=etree.Element('{'+W+'}rPr');fo=etree.SubElement(pr,'{'+W+'}rFonts');fo.set('{'+W+'}ascii','Cambria Math');fo.set('{'+W+'}hAnsi','Cambria Math');r.append(pr)
    t=el('t');t.text=text;r.append(t);return r
CMDS={'alpha':'α','beta':'β','gamma':'γ','delta':'δ','Delta':'Δ','epsilon':'ε','varepsilon':'ε','theta':'θ','vartheta':'ϑ','Omega':'Ω','lambda':'λ','mu':'μ','nu':'ν','xi':'ξ','rho':'ρ','sigma':'σ','Sigma':'Σ','tau':'τ','phi':'φ','pi':'π','Pi':'Π','eta':'η','kappa':'κ','chi':'χ','partial':'∂','infty':'∞','approx':'≈','propto':'∝','in':'∈','notin':'∉','le':'≤','ge':'≥','times':'×','cdot':'·','pm':'±','to':'→','perp':'⊥','ell':'ℓ','forall':'∀','leftarrow':'←','Longrightarrow':'⇒','rightleftharpoons':'⇌','quad':'  ','qquad':'    ','langle':'⟨','rangle':'⟩','Vert':'‖','vert':'|','prime':'′'}
CMDS.update({"Theta": "Θ", "omega": "ω", "Phi": "Φ"})
class Parser:
    def __init__(self,s):self.s=s;self.i=0
    def skip(self):
        while self.i<len(self.s) and self.s[self.i]==' ':self.i+=1
    def group(self):
        self.skip()
        if self.i<len(self.s) and self.s[self.i]=='{':self.i+=1;return self.seq('}')
        return [self.atom()]
    def literal_group(self):
        self.skip();assert self.s[self.i]=='{';self.i+=1;start=self.i;depth=1
        while depth:
            if self.s[self.i]=='{':depth+=1
            if self.s[self.i]=='}':depth-=1
            self.i+=1
        return self.s[start:self.i-1]
    def script(self,base,sub=None,sup=None):
        if sub is not None and sup is not None:return el('sSubSup',el('e',base),el('sub',sub),el('sup',sup))
        if sub is not None:return el('sSub',el('e',base),el('sub',sub))
        if sup is not None:return el('sSup',el('e',base),el('sup',sup))
        return base
    def seq(self,end=None):
        out=[]
        while self.i<len(self.s):
            if end and self.s[self.i]==end:self.i+=1;break
            if self.s[self.i]==' ':self.i+=1;continue
            base=self.atom();sub=sup=None
            while self.i<len(self.s) and self.s[self.i] in '_^':
                op=self.s[self.i];self.i+=1;v=self.group()
                if op=='_':sub=v
                else:sup=v
            out.append(self.script(base,sub,sup))
        return out
    def atom(self):
        if self.i>=len(self.s):return run('')
        c=self.s[self.i];self.i+=1
        if c=='{':return el('box',el('e',self.seq('}')))
        if c!='\\':
            if c.isdigit():
                match=re.match(r'[0-9]+(?:\.[0-9]*)?',self.s[self.i-1:])
                self.i=self.i-1+len(match.group())
                return run(match.group())
            return run('−' if c=='-' else c)
        if self.i<len(self.s) and not self.s[self.i].isalpha():
            c=self.s[self.i];self.i+=1;return run(' ' if c in ',;:!' else '‖' if c=='|' else c)
        m=re.match('[A-Za-z]+',self.s[self.i:]);cmd=m.group();self.i+=len(cmd)
        if cmd in CMDS:return run(CMDS[cmd])
        if cmd=='frac':return el('f',el('num',self.group()),el('den',self.group()))
        if cmd=='sqrt':return el('rad',el('radPr',el('degHide',val='1')),el('deg'),el('e',self.group()))
        if cmd in ['hat','bar','tilde']:
            return el('acc',el('accPr',el('chr',val={'hat':'̂','bar':'̄','tilde':'̃'}[cmd])),el('e',self.group()))
        if cmd in ['text','operatorname','mathrm']:return run(self.literal_group(),plain=True)
        if cmd=='mathcal':
            value=self.literal_group()
            return run({'F':'ℱ','J':'𝒥','P':'𝒫','B':'ℬ','G':'𝒢'}.get(value,value))
        if cmd in ['mathbf','mathbb']:return el('box',el('e',self.group()))
        if cmd in ['ln','log','exp','min','max','diag','argmin']:return run(cmd,plain=True)
        if cmd in ['int','sum']:
            sub=[];sup=[]
            while self.i<len(self.s) and self.s[self.i] in '_^':
                op=self.s[self.i];self.i+=1;v=self.group()
                if op=='_':sub=v
                else:sup=v
            self.skip();body=self.group() if self.i<len(self.s) and self.s[self.i]=='{' else []
            return el('nary',el('naryPr',el('chr',val='∫' if cmd=='int' else '∑'),el('limLoc',val='subSup'),el('grow',val='1'),el('subHide',val='0' if sub else '1'),el('supHide',val='0' if sup else '1')),el('sub',sub),el('sup',sup),el('e',body))
        if cmd in ['left','right','big','Big']:return self.atom()
        raise ValueError('Unsupported math command '+cmd)
def omml(s):return el('oMath',Parser(s).seq())
