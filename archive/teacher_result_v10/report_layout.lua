-- Dedicated cover page and numbered display equations for PDF/LaTeX and Word.
local function xml(text)
  return text:gsub('&', '&amp;'):gsub('<', '&lt;'):gsub('>', '&gt;')
end

local function tex(text)
  return text:gsub('([%%&#_{}])', '\\%1')
end

local function meta_text(meta, key)
  return meta[key] and pandoc.utils.stringify(meta[key]) or ''
end

local function cover(meta)
  local title = meta_text(meta, 'title')
  local subtitle = meta_text(meta, 'subtitle')
  local student = meta_text(meta, 'student-id')
  local name = meta_text(meta, 'student-name')
  local date = meta_text(meta, 'date'):gsub('^記録整理日：', '')
  if FORMAT:match('latex') then
    local heading = tex(title):gsub('による', 'による\\par ')
    return pandoc.RawBlock('latex', [[
\hypersetup{pageanchor=false}
\begin{titlepage}
\centering
\vspace*{22mm}
{\Large ]] .. tex(subtitle) .. [[\par}
\vspace{24mm}
\rule{0.88\linewidth}{0.7pt}\par
\vspace{9mm}
{\Huge\bfseries ]] .. heading .. [[\par}
\vspace{9mm}
\rule{0.88\linewidth}{0.7pt}\par
\vfill
\renewcommand{\arraystretch}{1.6}
{\large
\begin{tabular}{rl}
学籍番号 & ]] .. tex(student) .. [[ \\
氏名 & ]] .. tex(name) .. [[ \\
作成日 & ]] .. tex(date) .. [[
\end{tabular}
}
\vspace*{25mm}
\end{titlepage}
\hypersetup{pageanchor=true}
\setcounter{page}{1}
]])
  elseif FORMAT == 'docx' then
    local function paragraph(text, size, before, after, bold)
      return '<w:p><w:pPr><w:jc w:val="center"/>' ..
        '<w:ind w:firstLine="0"/><w:spacing w:before="' .. before .. '" w:after="' .. after .. '"/></w:pPr>' ..
        '<w:r><w:rPr><w:sz w:val="' .. size .. '"/>' .. (bold and '<w:b/>' or '') ..
        '</w:rPr><w:t xml:space="preserve">' .. xml(text) .. '</w:t></w:r></w:p>'
    end
    local first, second = title:match('^(.-による)(.*)$')
    first, second = first or title, second or ''
    return pandoc.RawBlock('openxml',
      paragraph(subtitle, 28, 1200, 1100, false) ..
      paragraph(first, 48, 0, 160, true) .. paragraph(second, 48, 0, 2000, true) ..
      paragraph('学籍番号　' .. student, 24, 0, 200, false) ..
      paragraph('氏名　' .. name, 24, 0, 200, false) ..
      paragraph('作成日　' .. date, 24, 0, 0, false) ..
      '<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
  end
end

function Pandoc(doc)
  local numbers = {}
  local count = 0
  doc = doc:walk({Div = function(div)
    if not div.classes:includes('numbered-equation') then return nil end
    count = count + 1
    numbers[div.identifier] = count
    local math = div.content[1] and div.content[1].content[1]
    if not math or math.t ~= 'Math' or math.mathtype ~= 'DisplayMath' then
      error('A numbered-equation div must contain one display equation: ' .. div.identifier)
    end
    if FORMAT:match('latex') then
      local expression = math.text:match('^%s*(.-)%s*$')
      return pandoc.RawBlock('latex', '\\begin{equation}\\label{' .. div.identifier .. '}\n' ..
        expression .. '\n\\end{equation}')
    elseif FORMAT == 'docx' then
      local table = pandoc.read('| L | C | R |\n|:---|:---:|---:|\n| | X | N |', 'markdown').blocks[1]
      table.identifier = div.identifier
      table.head.rows = {}
      table.colspecs = {{pandoc.AlignDefault,0.05},{pandoc.AlignCenter,0.90},{pandoc.AlignRight,0.05}}
      table.bodies[1].body[1].cells[2].contents = {pandoc.Para({math})}
      table.bodies[1].body[1].cells[3].contents = {pandoc.Para({pandoc.Str('(' .. count .. ')')})}
      return table
    end
    return div
  end})
  doc = doc:walk({Link = function(link)
    local id = link.target:match('^#(eq:.*)$')
    if id then
      if not numbers[id] then error('Unknown equation reference: ' .. id) end
      link.content = {pandoc.Str('(' .. numbers[id] .. ')')}
      return link
    end
  end})
  local page = cover(doc.meta)
  if page then
    doc.blocks:insert(1, page)
    doc.meta.pdftitle = doc.meta.title
    doc.meta.title, doc.meta.subtitle, doc.meta.author, doc.meta.date = nil, nil, nil, nil
  end
  return doc
end
