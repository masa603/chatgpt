-- Center numbered captions without changing report prose or automatic numbering.
function Para(el)
  local text = pandoc.utils.stringify(el)
  if not (text:match('^図%d+　') or text:match('^表%d+　')) then
    return nil
  end
  if FORMAT:match('latex') then
    return {
      pandoc.RawBlock('latex', '\\begingroup\\centering\\setlength{\\parindent}{0pt}'),
      el,
      pandoc.RawBlock('latex', '\\par\\endgroup')
    }
  elseif FORMAT == 'docx' then
    local escaped = text:gsub('&', '&amp;'):gsub('<', '&lt;'):gsub('>', '&gt;')
    return pandoc.RawBlock('openxml',
      '<w:p><w:pPr><w:jc w:val="center"/>' ..
      '<w:ind w:left="0" w:right="0" w:firstLine="0"/></w:pPr>' ..
      '<w:r><w:t xml:space="preserve">' .. escaped .. '</w:t></w:r></w:p>')
  end
end
