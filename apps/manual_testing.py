# ruff: noqa: RUF001
"""Loopback-only owner workbench. Raw candidates are never customer-service answers."""

import json

import streamlit as st

from supportpilot.common.config import find_project_root
from supportpilot.service.manual_testing import InspectionRequest, ManualWorkbench

st.set_page_config(page_title="SupportPilot · LLM", page_icon="🔬", layout="wide")
if st.get_option("server.address") != "127.0.0.1":
    st.error("Стенд разрешён только на 127.0.0.1. См. RUNBOOK, раздел 18.")
    st.stop()
st.title("SupportPilot · проверка реальной LLM")
st.warning(
    "Экспериментальный ответ модели может быть ошибочным. Семантика: NOT_EVALUATED. "
    "Ручная проверка не является независимой квалификацией 9Q2/9Q3."
)
st.caption(
    "Выбранная Qwen3-0.6B + LoRA · локальная NF4 · действующая KB. "
    "Основной сервис не вызывается. Заказы, платежи и очередь не изменяются."
)


@st.cache_resource
def workbench() -> ManualWorkbench:
    return ManualWorkbench(find_project_root())


language = st.sidebar.selectbox("Язык ответа / Жауап тілі / Reply language", ["ru", "kk", "en"])
st.sidebar.info(
    "Пишите на RU, KK, EN или смешанном RU–KK. Для смешанного вопроса выберите язык ответа. "
    "Каждый вопрос независим: история показана только для сравнения, в prompt не попадает."
)
st.sidebar.caption(
    "Используйте только вымышленные данные. Вопросы, ответы и источники сохраняются локально. "
    "Данные заказов и остатки не подключены; модель не может выполнить действие. "
    "Первый запрос загружает модели; предел ожидания — 180 секунд."
)
if "inspections" not in st.session_state:
    st.session_state.inspections = []
if st.sidebar.button("Очистить экран"):
    st.session_state.inspections = []
    st.rerun()

query = st.chat_input("Введите вопрос / Сұрақ / Question", max_chars=2000)
if query:
    with st.spinner("Поиск источников и реальная генерация LLM…"):
        try:
            result = workbench().inspect(InspectionRequest(query=query, language=language))
            st.session_state.inspections.append(result)
            st.session_state.inspections = st.session_state.inspections[-20:]
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))

for item in st.session_state.inspections:
    with st.chat_message("user"):
        st.text(item["request"]["query"])
    with st.chat_message("assistant"):
        st.caption(f"Ответ модели · {item['request']['language']} · {item['status']}")
        if item["status"] != "COMPLETE":
            st.error(
                "Попытка не завершена: "
                + str(item.get("error", "UNKNOWN"))
                + ". Следующий запрос автоматически перезапустит локальный worker."
            )
        else:
            checks = item["checks"]
            if checks["candidate"] is not None:
                st.text(checks["candidate"]["answer"])
            else:
                st.error(
                    "Модель вернула невалидный JSON. Ниже показан исходный вывод без исправлений."
                )
                st.text(item["raw"])
            sources, diagnostics, raw = st.tabs(
                ["Источники в prompt", "Проверки", "Сырой вывод и prompt"]
            )
            with sources:
                context = item["context"]
                st.caption(f"KB: {context['kb_version']} · Index: {context['index_version']}")
                if not context["evidence"]:
                    st.warning("Источников нет. Модель получила пустой набор доказательств.")
                for source in context["evidence"]:
                    chunk = source["chunk"]
                    with st.expander(f"{chunk['doc_id']} · {source['chunk_id']}"):
                        st.text(chunk["text"])
                        st.json(
                            {
                                key: chunk[key]
                                for key in (
                                    "source_path",
                                    "locator",
                                    "checksum",
                                    "effective_from",
                                    "effective_to",
                                )
                            }
                        )
            with diagnostics:
                st.text("Семантическая поддержка: UNKNOWN · качество NOT_EVALUATED")
                st.text("JSON: " + checks["schema"])
                st.text("Диагностика языка ответа: " + str(checks["detected_reply_language"]))
                st.text(
                    "Проблемы: " + (", ".join(checks["deterministic_problems"]) or "не найдены")
                )
                st.caption(
                    "JSON, язык, совпадение чисел и цитат не доказывают правильность. "
                    "Допуск фактического ответа: закрыт. Сервис и инструменты не вызывались."
                )
                st.json(item.get("resources", {}))
            with raw:
                st.text(item["raw"])
                st.json(item["messages"])
                st.json(item["identity"])
        st.caption(f"Попытка: {item['attempt']} · {item['wall_seconds']:.1f} с")
        st.download_button(
            "Скачать запись проверки",
            json.dumps(item, ensure_ascii=False, indent=2),
            file_name=f"inspection-{item['attempt']}.json",
            mime="application/json",
            key=item["attempt"],
        )
