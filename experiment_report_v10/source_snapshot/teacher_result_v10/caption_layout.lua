-- Center captions and separate figure captions from the following prose.
function Para(el)
  local text = pandoc.utils.stringify(el)
  local is_figure = text:match('^図%d+　') ~= nil
  if not (is_figure or text:match('^表%d+　')) then
    return nil
  end
  if FORMAT:match('latex') then
    return {
      pandoc.RawBlock('latex', '\\begingroup\\centering\\setlength{\\parindent}{0pt}'),
      el,
      pandoc.RawBlock('latex', '\\par\\endgroup' ..
        (is_figure and '\\vspace*{12pt}' or ''))
    }
  elseif FORMAT == 'docx' then
    local escaped = text:gsub('&', '&amp;'):gsub('<', '&lt;'):gsub('>', '&gt;')
    return pandoc.RawBlock('openxml',
      '<w:p><w:pPr><w:jc w:val="center"/>' ..
      '<w:ind w:left="0" w:right="0" w:firstLine="0"/>' ..
      (is_figure and '<w:spacing w:after="240" w:afterAutospacing="0"/><w:keepLines/>' or '') ..
      '</w:pPr>' ..
      '<w:r><w:t xml:space="preserve">' .. escaped .. '</w:t></w:r></w:p>')
  end
end
