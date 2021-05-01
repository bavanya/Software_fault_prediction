% Style file for LaTeX2e
%   - for use with documentclass report
%     \usepackage [fancyhdr]{McECEThesis}

% $Id: McECEThesis.sty,v 1.9 2000/03/26 12:46:34 pkabal Exp $

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{csethesis}[2007/04/09]

%\PassOptionsToClass{12pt}{report}
\PassOptionsToClass{twoside,12pt}{report}

\LoadClass{report}

%------------------ Defining the most common environments ---------------

%\newtheorem{theorem}{Theorem:}
\newtheorem{invar}{Invariant:}
%\newtheorem{lemma}[theorem]{Lemma:}
%\newtheorem{cor}[theorem]{Corollary:}
\newtheorem{obs}{Observation:}
\newtheorem{conj}{Conjecture:}
%\newtheorem{prop}[theorem]{Proposition:}

\def\QED{\ensuremath{{\square}}}
\def\markatright#1{\leavevmode\unskip\nobreak\quad\hspace*{\fill}{#1}}
\newenvironment{proof}
  {\begin{trivlist}\item[\hskip\labelsep{\bf Proof:}]}
    {\markatright{\QED}\end{trivlist}}

%------------------------------------------------------------------------

%\setlength{\textheight}{10.0in}
%\setlength{\textwidth}{6.500in}
%\setlength{\topmargin}{-0.900in}
%\setlength{\oddsidemargin}{-0.125in}
%\setlength{\evensidemargin}{-0.375in}


\newtoks\phdtitle
\newtoks\name
\newtoks\rollno
\newtoks\email
\newtoks\guide

\newif\if@fancyhdr
\DeclareOption{fancyhdr}{\@fancyhdrtrue}
\DeclareOption{plainhdr}{\@fancyhdrfalse}

\ExecuteOptions{plainhdr}
\ProcessOptions

%================== page layout
\newdimen \insidemargin
\newdimen \outsidemargin
\newdimen \abovemargin
\newdimen \belowmargin

%===== page layout defined by 4 dimensions, others are calculated from these
% Define the side margins for a right-side page
\insidemargin = 1.1in       % Actual space
\outsidemargin =1.1in

% Above margin is space above the header
% Below margin is space below footer
\abovemargin =0.75in        % Actual space
\belowmargin =0.75in

%=====

% Leave more space for the header (originally 12pt)
\headheight =12pt

% Other parameters retain their default values (values for 12pt documents)
% \headsep - space from bottom of header to top of text (25pt)
% \footskip - distance from the bottom of text to bottom of footer (30pt)

% The rest of the dimensions are calculated from the above values
% (compensation for TeX 1in margin offsets is done here)
\textwidth = \paperwidth
  \advance\textwidth by -\insidemargin
  \advance\textwidth by -\outsidemargin
\textheight = 11.4in
  \advance \textheight by -\abovemargin
  \advance \textheight by -\headheight
  \advance \textheight by -\headsep
  \advance \textheight by -\belowmargin
  \advance \textheight by -\footskip
\topmargin = \abovemargin
  \advance \topmargin by -1in

\oddsidemargin = \insidemargin
  \advance\oddsidemargin by -1in
\if@twoside
  \evensidemargin = \outsidemargin
    \advance\evensidemargin by -1in
\else
  \evensidemargin = \oddsidemargin
\fi

%========= Redefine \cleardoublepage to use pagestyle plain
% Notes:
%  \include xxx
%      expands to \clearpage \input xxx.tex \clearpage
%  \chapter
%      expands to \clearpage  OR \cleardoublepage
%                 \thispagestyle{plain}
%                  ...
% \cleardoublepage is modified to output an empty page with pagestyle plain
% for headings and footers.
\renewcommand{\cleardoublepage}{%
  \clearpage
  \if@twoside
    \ifodd\c@page
    \else
      \hbox{}\thispagestyle{plain}\newpage
      \if@twocolumn \hbox{}\newpage \fi
    \fi
  \fi}

%========= Redefine \thebibliography to put a entry in the TOC
%========= Change the name from "Bibliography" to "References"
\renewcommand\bibname{References}
\let\SV@thebibliography\thebibliography
\renewcommand{\thebibliography}[1]{%
  \if@twoside
    \clearpage
   % \cleardoublepage
  \else
    \clearpage
  \fi
  \addcontentsline{toc}{chapter}{\bibname}
  \SV@thebibliography{#1}}

%========= Dummy definitions if datestamp package is not loaded
\ifx\datestampbox\@undefined
  \let\datestampbox=\relax
  \def\rcs #1{}
\fi

%========= Fancy headings setup
% Single-sided printing
%   header:   3.2 Section        5
%             --------------------
% Double-sided printing
%   header:   4            Chapter
%   even      --------------------
%
%   header:   3.2 Section        5
%   odd       --------------------
%

\if@fancyhdr
  \RequirePackage{fancyhdr}
  \pagestyle{fancy}
  \if@twoside
    \renewcommand{\chaptermark}[1]{\markboth{#1}{}}
    \renewcommand{\sectionmark}[1]{\markright{\thesection\ #1}}
    \fancyhf{}          % clear all fields
    \fancyhead[LE,RO]{\bfseries \thepage}
    \fancyhead[LO]{\bfseries \nouppercase{\rightmark}}
    \fancyhead[RE]{\bfseries \nouppercase{\leftmark}}
    \fancyfoot[LO,RE]{\datestampbox}
  \else
    \renewcommand{\chaptermark}[1]{\markright{\thechapter\ #1}}
    \renewcommand{\sectionmark}[1]{}
    \fancyhf{}          % clear all fields
    \fancyhead[RE,RO]{\bfseries \thepage}
    \fancyhead[LE,LO]{\bfseries \nouppercase{\rightmark}}
    \fancyfoot[LO,RE]{\datestampbox}
  \fi

  \renewcommand{\footrulewidth}{0pt}
  \renewcommand{\headrulewidth}{0.4pt}
  \def\headrule{{\headrulehook
                 \hrule height\headrulewidth width\headwidth
                 \vskip-\headrulewidth}}
  \def\headrulehook{}
% Notes:
%  - \nouppercase is to knock out the hardwired uppercase in
%    contents and bibliography
%  - \headrulehook can used for colour, e.g. \def\headrulehook{color{red}}

% Redefine plain style (used by \chapter) to just a page number at the top
  \fancypagestyle{plain}{%
    \fancyhf{}          % clear all fields
    \fancyhead[LE,RO]{\bfseries \thepage}%
    \fancyfoot[LO,RE]{\datestampbox}}

\fi

%================== title page

% Date set as "August 1996"
\def\Month{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi}
\date{\Month\ \number\year}

% Organization
\newcommand\organization[1]{\def\@organization{#1}}
\def\@organization{\@latex@warning@no@line{No \noexpand\organization given}}

% Note
\newcommand\note[1]{\def\@note{#1}}
\def\@note{\@latex@warning@no@line{No \noexpand\note given}}

% Redefine the \maketitle macro (separate page) with new layout
% This is a simplified version of the LaTeX \maketitle macro, since we
% need only worry about a single author and a single institution.
\renewcommand\maketitle{\begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \let \footnote \thanks

% Reset the page style to empty
  \pagestyle{empty}

% -- vertical space --
% Title with \LARGE \bfseries
% Author with \Large \itshape
% Organization with \large
% Date with \large
% -- vertical space --
% Note

  \begin{center}%
    \null\vfil \vskip 60\p@ % vertical centering (extra 60pt at top)
    {\LARGE \bfseries \@title \par}%
    \vskip 10ex%
   {\Large \itshape \@author}%
    \vskip 3ex%
    {\large
     \lineskip 0.5em%
     \begin{tabular}[t]{c}%
       \@organization
     \end{tabular}\par}%
    \vskip 3ex%
    {\large \@date \par}%
    \vskip 5em%
    \vfil\null          % vertical centering
    \@note
  \end{center}\par
  \@thanks

  \end{titlepage}%

% Reset macros
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}

%================== general changes
% Smaller section and subsection heads, ragged right
% [classes.dtx]
\def\veryraggedright{\advance\rightskip by 0pt plus 1fil\relax}
\renewcommand\section{\@startsection {section}{1}{\z@}%
    {-3.5ex \@plus -1ex \@minus -.2ex}%
    {2.3ex \@plus.2ex}%
    {\normalfont\large\bfseries\veryraggedright}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
    {-3.25ex\@plus -1ex \@minus -.2ex}%
    {1.5ex \@plus .2ex}%
    {\normalfont\bfseries\veryraggedright}}

% Placement of figures and tables htbp instead of tbp
% Figure name, Table name in bold
% [classes.dtx]
\renewcommand\fps@figure{htbp}
\renewcommand\fps@table{htbp}
\renewcommand\fnum@figure{{\reset@font\bfseries\figurename~\thefigure}}
\renewcommand\fnum@table{{\reset@font\bfseries\tablename~\thetable}}


% Less space between the number and the section title (1em -> 0.5em)
% [ltsect.dtx]
\renewcommand\@seccntformat[1]{\csname the#1\endcsname\hskip 0.5em}

% Float captions
% - Prove a \Tcaption for "top" captions
% - Remove ":" separating the number from the caption
% - Typeset the caption in \small
% - Set the caption in a narrower (80%) parbox
% [classes.dtx]
% Note: use "\hbox to" rather than "\hb@xt@" for older latex2e's
\def\captiontextfraction{0.8}
\def\captionfontsize{\small}
\newdimen\aboveTcaptionskip \aboveTcaptionskip=\belowcaptionskip
\newdimen\belowTcaptionskip \belowTcaptionskip=0.5\abovecaptionskip
\def\Tcaption{%
  \abovecaptionskip=\aboveTcaptionskip
  \belowcaptionskip=\belowTcaptionskip
  \caption}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{\captionfontsize #1\quad #2}%
  \ifdim \wd\@tempboxa >\captiontextfraction\hsize
    \hbox to \hsize{\hfil%
     \parbox{\captiontextfraction\hsize}{\captionfontsize #1\quad #2}%
    \hfil}%
  \else
    \global \@minipagefalse
    \hbox to \hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}

% Figure name becomes "Fig." (was "Figure")
\renewcommand\figurename{Fig.}





