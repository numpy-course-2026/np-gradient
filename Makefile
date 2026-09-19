# Проверки проекта «Градиент».
#   make check        все тикеты
#   make check-grd2   один тикет
#   make update       забрать новые тикеты и проверки из шаблона курса

check:
	@bash checks/run.sh

check-%:
	@bash checks/run.sh $*

update:
	@bash checks/update.sh

.PHONY: check update
