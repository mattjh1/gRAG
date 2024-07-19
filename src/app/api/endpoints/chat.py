from fastapi import APIRouter

router = APIRouter()


@router.post("/")
async def test_endpoint(
    request: str,
):

    return {"response": "hello world"}
