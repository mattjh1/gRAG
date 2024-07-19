from loguru import logger

from app.util.chains import get_rag_chain

if __name__ == "__main__":
    original_query = "tell me about count dracula"
    chain = get_rag_chain()
    logger.info(
        chain.invoke(
            {"question": original_query},
        )
    )
