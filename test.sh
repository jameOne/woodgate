coverage run -m pytest ./woodgate/build/*
coverage html
coverage run -m pytest ./woodgate/model/*
coverage html
coverage run -m pytest ./woodgate/transfer/*
coverage html
coverage run -m pytest ./woodgate/tuning/*
coverage html
coverage run -m pytest ./woodgate/woodgate_*
coverage html

