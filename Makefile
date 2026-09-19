# Проверки проекта «Градиент».
#   make check        все тикеты
#   make check-grd2   один тикет

check:
	@bash checks/run.sh

check-%:
	@bash checks/run.sh $*

.PHONY: check
