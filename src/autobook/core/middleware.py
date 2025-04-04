import logging
import time

from fastapi import Request

from autobook.core.env import Env
from autobook.core.logger import setup_logger

env = Env()
logger = setup_logger("AutoBook Middleware", level=env.LOGGER_LEVEL)


class TimeLogMiddleware:
    def __init__(
        self,
        some_attribute: str,
    ):
        self.some_attribute = some_attribute

    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        end_time = time.time()

        execution_time = f"{end_time - start_time:.4f}"

        response.headers["X-Process-Time"] = execution_time
        logger.info(
            f"'{request.method} {request.url.path}' - execution time: {execution_time} s"
        )
        return response
