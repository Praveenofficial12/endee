import json
from pydantic import BaseModel, ValidationError

class AnalyzeResponse(BaseModel):
    explanation: str
    fix: str
    corrected_code: str

# Let's say user input triggers heuristic
user_lang = "Java"
user_input = """import java.util.Scanner;
public class PalindromeNumber {
"""

result = {
    "explanation": f"[Heuristic Analysis — {user_lang}]\n\nThe code contains an opening curly brace '{{' but is missing the corresponding closing brace '}}'.",
    "fix": "Add a closing curly brace '}' to properly close the code block.",
    "corrected_code": f'// Missing closing brace:\n{user_input}\n}}'
}

print(result)
try:
    print(AnalyzeResponse(**result))
    print("ALL OK")
except Exception as e:
    print("FAILED")
    print(e)
