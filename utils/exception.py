from typing import Any
import sys
import traceback


class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: tuple):
        super().__init__(error_message)
        self.error_message = self._get_detailed_error_message(error_message, error_detail)

    def _get_detailed_error_message(self, error_message: str, error_detail: tuple) -> str:
        exc_type, exc_obj, exc_tb = error_detail
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error in {file_name} at line {line_number}: {error_message}"

    def __str__(self):
        return self.error_message

