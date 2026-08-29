import re


def extract_math_answer(response):
    """
    从数学回答中提取最终答案。
    优先提取boxed内的内容，如果没有则提取其他格式的数学答案。
    如果没有匹配到任何模式，则提取最后一个数字。
    
    参数:
        response (str): 可能包含LaTeX的数学回答文本
    
    返回:
        str: 提取出的答案，已移除所有空格和前缀符号
    """
    
    boxed_pattern = r'\\boxed{([^{}]+(?:{[^{}]*})*[^{}]*)}'
    boxed_matches = re.findall(boxed_pattern, response)
    if boxed_matches:
        boxed_content = boxed_matches[-1].replace(" ", "")  # 如果有多个boxed，返回最后一个，并移除空格
        

        text_pattern = r'\\text{([^{}]+)}'
        text_matches = re.findall(text_pattern, boxed_content)
        if text_matches:
            # 如果找到 \text{}，则提取其中的内容
            boxed_content = text_matches[-1]

        # 查找等号后的内容
        if '=' in boxed_content:
            parts = boxed_content.split('=')
            # 取等号后的最后一部分
            result_part = parts[-1].strip()
            # 移除单位和多余符号
            # 匹配数字（整数或小数），允许负数
            number_match = re.search(r'[-+]?\d+(?:\.\d+)?', result_part)
            if number_match:
                boxed_content = number_match.group(0)

        # 如果boxed内容是星号或其他无效内容，跳过并使用其他规则
        if boxed_content not in ['*', '?', '...', '']:
            return clean_math_symbols(remove_end_token(boxed_content))
    
    # 检查坐标对或有序对
    coordinate_pattern = r'\(([^()]+)\)'
    coordinate_matches = re.findall(coordinate_pattern, response)
    if coordinate_matches and ',' in coordinate_matches[-1]:
        # 确保它是坐标对（包含逗号）
        answer = coordinate_matches[-1].replace(" ", "")
        return clean_math_symbols(remove_end_token(answer))
    
    # 查找常见的答案模式
    # 检查带有"="号的最终等式
    equation_pattern = r'=\s*([^=\n]+)(?:\s*[.,])?$'
    equation_matches = re.findall(equation_pattern, response)
    if equation_matches:
        equation_result = equation_matches[0].strip().replace(" ", "")
        
        equation_result = equation_result.replace(',', '')
        # --- 新增：从等式结果中提取数字 ---
        number_match = re.search(r'[-+]?\d+(?:\.\d+)?', equation_result)
        if number_match:
            # 如果找到数字，则返回数字
            answer = number_match.group(0)
        else:
            # 如果找不到数字，则返回原始等式结果
            answer = equation_result
            
        return clean_math_symbols(remove_end_token(answer))
    
    # 检查"答案是"或类似短语
    answer_phrase_pattern = r'(?:answer|result|value)[^\w\d\-]*(?:is|equals|=)[^\w\d\-]*([^.,\n]+)'
    answer_phrase_matches = re.findall(answer_phrase_pattern, response, re.IGNORECASE)
    if answer_phrase_matches:
        answer = answer_phrase_matches[-1].strip().replace(" ", "")
        return clean_math_symbols(remove_end_token(answer))
    
    # 提取最后一个数字（包括分数、小数和负数）
    number_pattern = r'[-+]?\d+(?:\.\d+)?|[-+]?\d+\/\d+'
    number_matches = re.findall(number_pattern, response)
    if number_matches:
        answer = number_matches[-1].replace(" ", "")
        return clean_math_symbols(remove_end_token(answer))
    
    # 检查特定的数学格式
    # 分数 - 改进处理方式
    fraction_pattern = r'\\frac{([^{}]+(?:{[^{}]*})*[^{}]*)}{([^{}]+(?:{[^{}]*})*[^{}]*)}'
    fraction_matches = re.findall(fraction_pattern, response)
    if fraction_matches:
        # 返回最后一个分数表达式
        numerator, denominator = fraction_matches[-1]
        answer = f"\\frac{{{numerator.replace(' ', '')}}}{{{denominator.replace(' ', '')}}}"
        return clean_math_symbols(remove_end_token(answer))
    
    # 区间
    interval_pattern = r'\([^{}]*,[^{}]*\)'
    interval_matches = re.findall(interval_pattern, response)
    if interval_matches:
        answer = f"({interval_matches[-1].replace(' ', '')})"
        return clean_math_symbols(remove_end_token(answer))
    
    # 平方根
    sqrt_pattern = r'\\sqrt{([^{}]+(?:{[^{}]*})*[^{}]*)}'
    sqrt_matches = re.findall(sqrt_pattern, response)
    if sqrt_matches:
        answer = f"\\sqrt{{{sqrt_matches[-1].replace(' ', '')}}}"
        return clean_math_symbols(remove_end_token(answer))

    
    # 如果没有匹配的模式，返回最后一行作为备选
    lines = response.strip().split('\n')
    answer = lines[-1].strip().replace(" ", "")
    return clean_math_symbols(remove_end_token(answer))


def extract_qa_answer(response):
    """
    从QA回答中提取True/False或Yes/No答案。
    
    参数:
        response (str): QA回答文本
    
    返回:
        str: 提取出的答案，标准化为"True"或"False"
    """
    # 首先检查boxed内容（最高优先级）
    boxed_pattern = r'\\boxed{([^{}]+(?:{[^{}]*})*[^{}]*)}'
    boxed_matches = re.findall(boxed_pattern, response)
    if boxed_matches:
        boxed_content = boxed_matches[-1].strip().lower()
        if any(word in boxed_content for word in ['true', 'yes', 'correct']):
            return "True"
        elif any(word in boxed_content for word in ['false', 'no', 'incorrect']):
            return "False"
    
    # 将文本转换为小写并移除空格
    text = response.lower().strip()
    
    # 检查True/False模式
    if re.search(r'\btrue\b', text) or re.search(r'\byes\b', text) or re.search(r'\bcorrect\b', text):
        return "True"
    elif re.search(r'\bfalse\b', text) or re.search(r'\bno\b', text) or re.search(r'\bincorrect\b', text):
        return "False"
    
    # 检查"答案是"或类似短语后面跟着的True/False
    answer_true_pattern = r'(?:answer|conclusion|result)[^\w\d\-]*(?:is|=)[^\w\d\-]*(true|yes|correct)'
    answer_false_pattern = r'(?:answer|conclusion|result)[^\w\d\-]*(?:is|=)[^\w\d\-]*(false|no|incorrect)'
    
    if re.search(answer_true_pattern, text, re.IGNORECASE):
        return "True"
    elif re.search(answer_false_pattern, text, re.IGNORECASE):
        return "False"
    
    # 如果没有明确的True/False指示，尝试分析整体语义
    positive_indicators = ['agree', 'support', 'confirm', 'right', 'valid', 'accurate']
    negative_indicators = ['disagree', 'reject', 'deny', 'wrong', 'invalid', 'inaccurate']
    
    positive_count = sum(1 for word in positive_indicators if word in text)
    negative_count = sum(1 for word in negative_indicators if word in text)
    
    if positive_count > negative_count:
        return "True"
    elif negative_count > positive_count:
        return "False"
    
    # 如果无法确定，默认返回False
    return "False"


def clean_math_symbols(text):
    """
    清理数学答案中的前缀和后缀符号
    移除货币符号、单位、LaTeX格式符号等
    
    参数:
        text (str): 需要清理的文本
        
    返回:
        str: 清理后的文本
    """
    if not text:
        return text
    
    # 移除前后的美元符号
    text = text.strip()
    
    # 处理LaTeX美元符号 \$ 
    text = text.replace('\\$', '')
    text = text.replace('\\%', '')
    text = text.replace('%', '')
    text = text.replace('\\#', '')
    text = text.replace('£', '')
    text = text.replace('€', '')
    
    units_to_remove = [
        '\\textdollar', 'kg', 'mg', 'μg', 'km', 'cm', 'ml', 'g', 'm'
    ]
    
    for unit in units_to_remove:
        text = text.replace(unit, '')

    text = re.sub(r'\\(-?\d)', r'\1', text)
    
    # 移除包装性的美元符号
    while text.startswith('$') and text.endswith('$') and len(text) > 1:
        text = text[1:-1].strip()
    
    # 移除前缀的美元符号
    while text.startswith('$'):
        text = text[1:].strip()
    
    # 移除后缀的美元符号
    while text.endswith('$'):
        text = text[:-1].strip()
    
    # 移除数字中的逗号分隔符（千位分隔符）
    # 但保留坐标对中的逗号，如 (3,4) 或分数中的逗号
    if not (text.startswith('(') and text.endswith(')') and text.count(',') == 1):
        # 使用正则表达式只移除数字中的千位分隔符逗号
        # 匹配形如 1,234 或 1,234.56 的模式
        # 检查是否是纯数字格式（可能包含千位分隔符）
        if re.match(r'^-?[\d,]+\.?\d*$', text.replace(' ', '')):
            # 移除所有逗号
            text = text.replace(',', '')
        elif re.match(r'^[\d,]+$', text.replace(' ', '')):
            # 纯整数格式，移除逗号
            text = text.replace(',', '')
        else:
            # 对于复杂表达式，只移除明显的千位分隔符
            # 保留可能是坐标对或其他用途的逗号
            text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
            text = re.sub(r'(\d),(\d{3})', r'\1\2', text)  # 可能需要多次替换
            text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    
    # 移除前缀的反斜线（但保留LaTeX命令）
    # 只移除单独的反斜线，不移除LaTeX命令如\frac, \sqrt等
    if text.startswith('\\') and not text.startswith('\\frac') and not text.startswith('\\sqrt') and not text.startswith('\\boxed'):
        # 检查是否是单独的反斜线或者不是LaTeX命令
        if len(text) == 1 or (len(text) > 1 and text[1] in ' \t\n'):
            text = text[1:].strip()
    
    # 移除前后的圆括号（如果不是坐标对）
    if text.startswith('(') and text.endswith(')') and ',' not in text:
        inner_text = text[1:-1].strip()
        # 确保括号是配对的，且内容不为空
        if inner_text and inner_text.count('(') == inner_text.count(')'):
            text = inner_text

    if re.match(r'^-?\d+\.0+$', text):
        # 匹配如 "32.00", "123.000", "-5.0" 等
        text = text.split('.')[0]
    
    # 移除前后多余的空白字符
    text = text.strip()
    
    return text


def remove_end_token(text):
    """
    移除文本中的<|end|>标记
    
    参数:
        text (str): 可能包含<|end|>标记的文本
        
    返回:
        str: 移除<|end|>标记后的文本
    """
    return text.replace("<|end|>", "")


def extract_answer(response, dataset_type="math", dataset_name="GSM8k", reference_answer=False):
    """
    根据数据集类型选择合适的提取器提取答案
    
    参数:
        response (str): 回答文本
        dataset_type (str): 数据集类型，可以是"math"或"qa"
    
    返回:
        str: 提取出的答案
    """

    # 首先检查boxed内容（最高优先级）
    if dataset_name == "GSM8k" and reference_answer:
        hash_pattern = r'####\s*([^\n]+)'
        hash_matches = re.findall(hash_pattern, response)
        if hash_matches:
            answer = hash_matches[-1].strip()
            return clean_math_symbols(remove_end_token(answer))
    
    answer = None
    if dataset_type.lower() == "math":
        answer = extract_math_answer(response)
    elif dataset_type.lower() == "qa":
        answer = extract_qa_answer(response)
    else:
        # 默认使用数学提取器
        answer = extract_math_answer(response)
    
    # 确保最终返回的答案也经过完整的清理流程
    return clean_math_symbols(remove_end_token(answer))