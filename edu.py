import streamlit as st
import sqlite3
import base64
import json
import time
import requests
import pandas as pd
import fitz  # PyMuPDF
import uuid
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import numpy as np

# =========================
# 1. 数据库设置：记录心理动态档案、学情历史与用户体系
# =========================
def init_db():
    conn = sqlite3.connect('xiewo_student_archive.db')
    c = conn.cursor()
    # 历史记录表：包含学生的心理画像
    c.execute('''
        CREATE TABLE IF NOT EXISTS student_profiles (
            student_id TEXT PRIMARY KEY,
            recent_emotion TEXT,  
            continuous_errors INTEGER,
            strength TEXT
        )
    ''')
    # 批改历史表，用于学情分析
    c.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            subject TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            score INTEGER,
            weak_points TEXT
        )
    ''')
    
    # 题库表：用于智能提取 PDF 题目
    c.execute('''
        CREATE TABLE IF NOT EXISTS question_bank (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            subject TEXT,
            source_pdf TEXT,
            page_num INTEGER,
            question_text TEXT,
            question_id TEXT
        )
    ''')
    # 用户表：用于模拟登录系统
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            role TEXT,
            student_id TEXT,
            real_name TEXT
        )
    ''')
    
    # 预设账号
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO users (username, password, role, student_id, real_name) VALUES ('stu1', '123', 'student', 'Stu-001', '张小亮')")
        c.execute("INSERT INTO users (username, password, role, student_id, real_name) VALUES ('stu2', '123', 'student', 'Stu-002', '李文文')")
        c.execute("INSERT INTO users (username, password, role, student_id, real_name) VALUES ('teacher', '123', 'teacher', '', '王老师')")
        
    conn.commit()
    conn.close()

# 自动存档与策略计算模块
def update_profile(student_id, extracted_emotion, correctness, subject, score, weak_points_list):
    conn = sqlite3.connect('xiewo_student_archive.db')
    c = conn.cursor()
    
    # 简单策略: 如果最近连续答错, 加深挫折判定
    c.execute('SELECT * FROM student_profiles WHERE student_id=?', (student_id,))
    data = c.fetchone()
    
    continuous_errors = 0 if correctness else 1
    if data and not correctness:
        continuous_errors = data[2] + 1
        
    c.execute('''
        INSERT INTO student_profiles (student_id, recent_emotion, continuous_errors, strength)
        VALUES (?, ?, ?, '发散思维')
        ON CONFLICT(student_id) DO UPDATE SET
        recent_emotion = excluded.recent_emotion,
        continuous_errors = excluded.continuous_errors
    ''', (student_id, extracted_emotion, continuous_errors))
    
    # 写入批改历史，支持学情看板
    weak_points_str = ",".join(weak_points_list) if weak_points_list else ""
    c.execute('''
        INSERT INTO evaluation_history (student_id, subject, score, weak_points)
        VALUES (?, ?, ?, ?)
    ''', (student_id, subject, score, weak_points_str))
    
    conn.commit()
    return data


import os
import streamlit as st

# 优先尝试从 st.secrets 获取，如果没有则降级为空字符串（部署时需在云端配置）
try:
    YOUR_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    YOUR_API_KEY = os.environ.get("GEMINI_API_KEY", "")

API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"


def get_real_vlm_analysis(image_bytes, history_profile_text, subject="数学", question_context=None):
    """
    真实的 Google Gemini 视觉大模型请求接口
    """
    img_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # 动态适应不同学科的要求
    subject_focus = ""
    if subject == "数学":
        subject_focus = "请识别手写公式、计算过程，给出‘过程分’，提取具体的薄弱知识点，并给出极其详尽、毫无遗漏的正确推导步骤。"
    elif subject == "计算机":
        subject_focus = "请识别手写/打印的代码片段，找出语法错误或逻辑漏洞，给出‘代码规范度分’和‘逻辑正确性分’，提取薄弱知识点，并给出详细的正确代码实现及修改原因。"
        
    context_injection = f"\n【特别注意】这是一道已知原题的解答批改，原题干如下：\n{question_context}\n请严格比对学生的解答与原题要求是否相符。\n" if question_context else ""
    
    system_prompt = f"""
    您是一台极其严谨的自适应教师批阅引擎。
    当前处理科目：【{subject}】。
    {context_injection}
    【核心任务】：解答必须详细、不遗漏、不凭空假设。您的主线任务是提供绝对正确、步骤详尽的学术解答。
    【辅助任务】：当前请先做学生近期的基本状态预输入研判，其历史系统存库数据状态为: {history_profile_text if history_profile_text else '刚入学, 尚无特征'}。您需要根据该学生的倾向（如急躁、挫折），在保证解答绝对严谨详细的前提下，稍微调整您的【语气和沟通方式】。

    请对以下提供给您的该生【作业照片】完成以下分析输出：
    1. 判断是否正确并找到根因。
    2. {subject_focus}
    3. 根据卷面涂改密集度给一个性格倾向提取(Emotional_Extract)。
    4. 反馈给数据库此人在当期的辅导基调(Adaptive_Tone_Selected)。
    5. 生成一段话术（generated_response）。**注意：这段话术必须包含极其详尽、一步一步的正确解题/改错过程，绝不可省略关键步骤，只需用符合该生性格倾向的语气包装即可。**

    ###强制要求：你必须且只能以严格合法的 JSON 纯数据格式响应。请确保在 `generated_response` 中所有的换行符(`\n`)和引号都被正确转义，绝不可破坏 JSON 结构！结构范例: 
    {{
       "visual_psych_scan": {{
          "handwriting_status": "发现xx位置有...",
          "emotional_inference": "xxx倾向...",
          "adapted_tone_selected": "该情况我正在采用了...基调引导"
       }},
       "academic_correction": {{
           "correctness": false, 
           "score": 60,
           "process_points": "公式列对得30分，计算错扣40分",
           "weak_knowledge_points": ["去括号符号变化", "粗心计算"],
           "root_cause": "漏标了括号..."
       }},
       "generated_response": "同学你好！[符合性格的简短开场]。下面是这道题完整且详细的正确解答过程：第一步... 第二步... 第三步... 因为[具体原因]，所以这里必须[具体操作]。请核对你的步骤..."
    }}
    """
    
    url = f"{API_ENDPOINT}?key={YOUR_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "parts": [
                    {"text": "照片分析开始："},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": img_b64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2, 
            "responseMimeType": "application/json"
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response_json = response.json()
        
        if 'candidates' in response_json:
            text_content = response_json['candidates'][0]['content']['parts'][0]['text'].strip()
            
            # 清理可能存在的 markdown 代码块包裹
            if text_content.startswith('```json'): text_content = text_content[7:]
            elif text_content.startswith('```'): text_content = text_content[3:]
            if text_content.endswith('```'): text_content = text_content[:-3]
            text_content = text_content.strip()
            
            try:
                parsed_result = json.loads(text_content) 
                return parsed_result
            except json.JSONDecodeError as e:
                raise Exception(f"大模型返回了格式损坏的 JSON 数据（如未转义的特殊符号）。解析错误: {str(e)}")
        else:
            raise Exception(f"API Error: {response_json}")
        
    except Exception as e:
        return {
           "visual_psych_scan": {
              "handwriting_status": f"真实接口请求遇阻...: {str(e)}",
              "emotional_inference": "请求受阻...",
              "adapted_tone_selected": "进入兜底响应..."
           },
           "academic_correction": {
               "correctness": False, 
               "score": 0,
               "process_points": "暂无",
               "weak_knowledge_points": ["网络请求受阻"],
               "root_cause": "API调用失败"
           },
           "generated_response": f"抱歉，真实的大模型接口连接失败了，请检查网络或 API Key。详细报错信息：{str(e)}"
        }

def extract_questions_from_page(img_b64, subject):
    """提取页面上的所有题目为 JSON 数组（单次调用，省额度）"""
    system_prompt = f"""
    您是一个专业的教育数据结构化提取引擎。当前提取科目：【{subject}】。
    请识别这张书页截图中的【全部】题目，从第一题到最后一题，一个不漏。
    【绝对禁止】中途停止、偷懒、省略！您必须从页面顶部扫描到页面底部，输出完整的题目列表。
    即使页面上有 30 道题，您也必须全部输出，不可以只输出前几道就停下来。
    忽略页眉页脚、非题目的讲解正文。
    ###强制要求：你必须且只能以严格合法的 JSON 纯数据格式响应。返回一个 JSON 数组(List)。
    请确保所有的换行符(\\n)和引号都被正确转义，绝不可破坏 JSON 结构！
    结构范例: 
    [
       {{ "q_num": 1, "content": "题干内容..." }},
       {{ "q_num": 2, "content": "第二题的内容..." }}
    ]
    """
    url = f"{API_ENDPOINT}?key={YOUR_API_KEY}"
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": "请提取这一页上的【全部】题目，从第一题到最后一题，一个不漏："}, {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}]}],
        "generationConfig": {
            "temperature": 0.1, 
            "responseMimeType": "application/json",
            "maxOutputTokens": 65536
        }
    }
    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=120)
        resp_json = response.json()
        if 'candidates' in resp_json:
            candidate = resp_json['candidates'][0]
            finish_reason = candidate.get('finishReason', '')
            
            # 检测是否因为输出过长被截断
            if finish_reason == 'MAX_TOKENS':
                st.toast("⚠️ 该页题目过多，输出被截断，部分题目可能丢失")
            
            import re
            txt = candidate['content']['parts'][0]['text'].strip()
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', txt)
            if json_match:
                txt = json_match.group(1)
            else:
                # 尝试找到 JSON 数组或对象
                arr_start = txt.find('[')
                obj_start = txt.find('{')
                if arr_start != -1 and (obj_start == -1 or arr_start < obj_start):
                    txt = txt[arr_start:txt.rfind(']')+1]
                elif obj_start != -1:
                    txt = txt[obj_start:txt.rfind('}')+1]
            parsed = json.loads(txt.strip())
            if isinstance(parsed, list):
                st.toast(f"本页提取到 {len(parsed)} 道题目")
                return parsed
            elif isinstance(parsed, dict):
                questions = parsed.get("questions", [])
                st.toast(f"本页提取到 {len(questions)} 道题目")
                return questions
        else:
            st.toast(f"提取出错: {resp_json.get('error', {}).get('message', '未知错误')}")
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        st.toast(f"⚠️ 该页 AI 返回的数据格式损坏，可能是输出被截断")
    except Exception as e:
        print(f"Extraction error: {e}")
        st.toast(f"提取时发生异常: {str(e)[:100]}")
    return []

# =========================
# 3. 前端交互大厅：打造展示炫酷视觉
# =========================
st.set_page_config(page_title="星芒 AI - 智能辅导", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* 极简主义设计 (Minimalism) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 增加呼吸感 (Whitespace) */
    .block-container {
        max-width: 95% !important;
    }
    
    /* 去除顶部大片留白 */
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 2rem !important;
    }
    
    /* 强制关闭所有默认的过渡动画，极速响应，去除“缓慢”的卡顿感 */
    * {
        animation-duration: 0.001s !important;
        transition-duration: 0.001s !important;
    }
    
    /* 去除不必要的分割线和边框 */
    hr {
        border-top: 1px solid rgba(200, 200, 200, 0.1);
        margin: 1.5rem 0;
    }
    
    /* 左对齐 tertiary 按钮内部元素 */
    button[kind="tertiary"] {
        justify-content: flex-start !important;
    }
    button[kind="tertiary"] div[data-testid="stMarkdownContainer"] p {
        text-align: left !important;
        margin: 0 !important;
    }
    button[kind="tertiary"] div {
        justify-content: flex-start !important;
    }
</style>
""", unsafe_allow_html=True)

init_db()

# 强制解除移动端/平板的缩放限制
import streamlit.components.v1 as components
components.html(
    """
    <script>
        var metas = window.parent.document.getElementsByTagName('meta');
        for (var i=0; i<metas.length; i++) {
            if (metas[i].name == "viewport") {
                metas[i].content = "width=device-width, initial-scale=1.0, maximum-scale=10.0, user-scalable=yes";
            }
        }
    </script>
    """,
    height=0,
    width=0,
)

# --- 登录模块 ---
if 'logged_in' not in st.session_state:
    if "auto_login_username" in st.query_params:
        _u = st.query_params["auto_login_username"]
        conn = sqlite3.connect('xiewo_student_archive.db')
        _c = conn.cursor()
        _c.execute("SELECT role, student_id, real_name FROM users WHERE username=?", (_u,))
        _user = _c.fetchone()
        conn.close()
        if _user:
            st.session_state['logged_in'] = True
            st.session_state['username'] = _u
            st.session_state['role'] = _user[0]
            st.session_state['student_id'] = _user[1]
            st.session_state['real_name'] = _user[2]
        else:
            st.session_state['logged_in'] = False
    else:
        st.session_state['logged_in'] = False

def check_login():
    u = st.session_state.login_u
    p = st.session_state.login_p
    conn = sqlite3.connect('xiewo_student_archive.db')
    c = conn.cursor()
    c.execute("SELECT role, student_id, real_name FROM users WHERE username=? AND password=?", (u, p))
    user = c.fetchone()
    conn.close()
    
    if user:
        st.query_params["auto_login_username"] = u
        st.session_state['logged_in'] = True
        st.session_state['username'] = u
        st.session_state['role'] = user[0]
        st.session_state['student_id'] = user[1]
        st.session_state['real_name'] = user[2]
    else:
        st.session_state['login_failed'] = True

login_placeholder = st.empty()

if not st.session_state.get('logged_in', False):
    with login_placeholder.container():
        st.title("星芒教具")
        
        if 'show_register' not in st.session_state:
            st.session_state['show_register'] = False
            
        if not st.session_state['show_register']:
            # 极简登录界面
            with st.form("login_form"):
                st.text_input("用户名", key="login_u")
                st.text_input("密码", type="password", key="login_p")
                st.form_submit_button("登录系统", on_click=check_login)
                
                if st.session_state.get('login_failed', False):
                    st.error("用户名或密码错误，请重试！")
                    st.session_state['login_failed'] = False
                        
            if st.button("没有账号？点击注册新账号"):
                st.session_state['show_register'] = True
                st.rerun()
                
        else:
            # 注册界面
            with st.form("reg_form"):
                reg_username = st.text_input("新建用户名")
                reg_password = st.text_input("设置密码", type="password")
                reg_realname = st.text_input("真实姓名")
                reg_role = st.selectbox("选择注册身份", ["学生", "教师"])
                reg_submitted = st.form_submit_button("立即注册")
                
                if reg_submitted:
                    if not reg_username or not reg_password or not reg_realname:
                        st.error("请完整填写所有信息！")
                    else:
                        conn = sqlite3.connect('xiewo_student_archive.db')
                        c = conn.cursor()
                        c.execute("SELECT * FROM users WHERE username=?", (reg_username,))
                        if c.fetchone():
                            st.error("该用户名已被占用，请换一个！")
                        else:
                            role_val = 'student' if reg_role == '学生' else 'teacher'
                            import uuid
                            student_id_val = f"Stu-{str(uuid.uuid4())[:4]}" if role_val == 'student' else ""
                            
                            c.execute("INSERT INTO users (username, password, role, student_id, real_name) VALUES (?, ?, ?, ?, ?)", 
                                      (reg_username, reg_password, role_val, student_id_val, reg_realname))
                            conn.commit()
                            st.success("注册成功！即将自动返回登录页面...")
                            time.sleep(1.5)
                            st.session_state['show_register'] = False
                            st.rerun()
                        conn.close()
                        
            if st.button("已有账号？返回登录"):
                st.session_state['show_register'] = False
                st.rerun()
else:
    # 已登录状态的主界面
    
    with st.sidebar:
        # A nice styled header for the system
        st.markdown("<h2 style='text-align: center; color: #4B4B4B;'>🌟 星芒教具</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888;'>智能学习辅助系统</p>", unsafe_allow_html=True)
        st.divider()
        
        # User profile block
        role_icon = "👨‍🎓" if st.session_state['role'] == 'student' else "👨‍🏫"
        role_display = "学生" if st.session_state['role'] == 'student' else "教师"
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #e9ecef;'>
            <div style='font-size: 40px; margin-bottom: 5px;'>{role_icon}</div>
            <div style='font-size: 18px; font-weight: bold; color: #343a40;'>{st.session_state['real_name']}</div>
            <div style='font-size: 14px; color: #6c757d; margin-top: 3px;'>{role_display} | {st.session_state['username']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.write("")
        
        # Subject selector
        st.markdown("### 控制面板")
        if st.session_state['role'] == 'student':
            global_subject = st.selectbox("选择学科", ["数学", "计算机"], help="切换学科将同步刷新上传区与学情统计区")
        else:
            global_subject = st.selectbox("全局监控学科", ["全部", "数学", "计算机"], help="切换学科将同步刷新全班监控看板")
            
        if 'last_subject' not in st.session_state:
            st.session_state['last_subject'] = global_subject
            
        if st.session_state['last_subject'] != global_subject:
            st.session_state['active_question_id'] = None
            st.session_state['active_folder'] = None
            st.session_state['ai_result'] = None
            st.session_state['last_subject'] = global_subject
            st.rerun()
            
        st.write("")
        st.write("")
        st.divider()
        
        if st.button("🚪 退出登录", use_container_width=True):
            st.query_params.clear()
            st.session_state.clear()
            st.rerun()
            
    # ---------------- 核心路由分发 ----------------
    if st.session_state['role'] == 'student':
        # --- 学生端视图 ---
        student_id_val = st.session_state['student_id']
        tab1, tab2, tab3 = st.tabs(["我的批改区", "我的学情统计", "智能刷题区"])
        
        with tab1:
            st.header(f"上传 {global_subject} 作业")
            
            uploaded_img = st.file_uploader("拍照或拖拽图片至此", type=['png','jpg','jpeg'])

            if uploaded_img:
                st.write("")
                if st.button(f"开始智能批改 {global_subject} 作业", type="primary", use_container_width=True):
                    subject_val = global_subject
                    bytes_data = uploaded_img.read()
                    
                    with st.container():
                        col1, col2 = st.columns([1, 1.2])
                        with col1:
                            st.image(uploaded_img, caption=f"{subject_val} 原卷")
                        
                        with col2:
                            st.subheader("批改与解析")
                            with st.spinner("AI 正在严谨分析您的作业，请稍候..."):
                                # --- 调用真实的 Gemini API ---
                                vlm_res = get_real_vlm_analysis(bytes_data, None, subject_val)
                                
                                # 更新数据库
                                update_profile(
                                    student_id_val, 
                                    vlm_res['visual_psych_scan']['emotional_inference'], 
                                    vlm_res['academic_correction']['correctness'],
                                    subject_val,
                                    vlm_res['academic_correction']['score'],
                                    vlm_res['academic_correction']['weak_knowledge_points']
                                )
                            
                            st.success("批改完成")
                            
                            met_c1, met_c2 = st.columns(2)
                            met_c1.metric("智能总评分", f"{vlm_res['academic_correction']['score']} 分")
                            
                            weak_points = vlm_res['academic_correction']['weak_knowledge_points']
                            met_c2.metric("核心薄弱点", f"{', '.join(weak_points) if weak_points else '无'}")
                            
                            st.markdown(f"**过程得分明细**：`{vlm_res['academic_correction']['process_points']}`")
                            
                            st.markdown(f"### 详细解析与推演:")
                            st.info(vlm_res['generated_response'])

        with tab2:
            st.header(f"个人 {global_subject} 知识点掌握情况")
            
            conn = sqlite3.connect('xiewo_student_archive.db')
            # 精准查询当前学生在选中科目的数据
            df = pd.read_sql_query(f"SELECT * FROM evaluation_history WHERE student_id='{student_id_val}' AND subject='{global_subject}'", conn)
            conn.close()
            
            if not df.empty:
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.subheader("我的高频易错点")
                    all_weak_points = []
                    for wp_str in df['weak_points'].dropna():
                        if wp_str:
                            all_weak_points.extend(wp_str.split(','))
                    
                    if all_weak_points:
                        wp_counts = pd.Series(all_weak_points).value_counts().reset_index()
                        wp_counts.columns = ['知识点', '错误频次']
                        st.bar_chart(data=wp_counts.set_index('知识点'), y_label="犯错次数", color="#FF4B4B")
                    else:
                        st.write("暂无薄弱点数据，表现很棒！")
                        
                with col_chart2:
                    st.subheader("我的学科历史得分")
                    # 为了展示折线图，把时间作为横坐标
                    if len(df) > 0:
                        df_line = df.copy()
                        df_line['timestamp'] = pd.to_datetime(df_line['timestamp'])
                        # 重置索引并pivot出漂亮的格式
                        chart_data = df_line.pivot_table(index='timestamp', columns='subject', values='score', aggfunc='mean')
                        st.line_chart(chart_data)
                    else:
                        st.write("暂无分数数据。")
                        
                st.subheader("我的详细批改记录")
                st.dataframe(df[['subject', 'score', 'weak_points', 'timestamp']], use_container_width=True)
                
                st.divider()
                if st.button("清空我的答题记录"):
                    conn = sqlite3.connect('xiewo_student_archive.db')
                    conn.execute("DELETE FROM evaluation_history WHERE student_id=?", (st.session_state['student_id'],))
                    conn.commit()
                    conn.close()
                    st.success("您的记录已清空！")
                    time.sleep(1)
                    st.rerun()
            else:
                st.info("尚未收到您的作业批改记录，请去【我的批改区】提交作业！")

        with tab3:
            if 'active_question_id' not in st.session_state:
                st.session_state['active_question_id'] = None
            if 'active_folder' not in st.session_state:
                st.session_state['active_folder'] = None
                
            if st.session_state['active_question_id'] is not None:
                # 沉浸式答题间模式：动态隐藏左侧边栏
                st.markdown("""
                    <style>
                    [data-testid="stSidebar"] { display: none !important; }
                    [data-testid="collapsedControl"] { display: none !important; }
                    </style>
                """, unsafe_allow_html=True)
                
                conn = sqlite3.connect('xiewo_student_archive.db')
                df_active = pd.read_sql_query(f"SELECT * FROM question_bank WHERE question_id='{st.session_state['active_question_id']}'", conn)
                conn.close()
                
                if df_active.empty:
                    st.session_state['active_question_id'] = None
                    st.rerun()
                else:
                    active_q = df_active.iloc[0]
                    active_pdf = active_q['source_pdf']
                    
                    conn = sqlite3.connect('xiewo_student_archive.db')
                    df_all_q = pd.read_sql_query(f"SELECT * FROM question_bank WHERE student_id='{student_id_val}' AND subject='{global_subject}' AND source_pdf='{active_pdf}' ORDER BY id ASC", conn)
                    conn.close()
                    
                    current_idx = df_all_q.index[df_all_q['question_id'] == active_q['question_id']].tolist()[0]
                    prev_q_id = df_all_q.iloc[current_idx - 1]['question_id'] if current_idx > 0 else None
                    next_q_id = df_all_q.iloc[current_idx + 1]['question_id'] if current_idx < len(df_all_q) - 1 else None
                    
                    col_back, col_space, col_prev, col_next = st.columns([1, 2, 1, 1])
                    with col_back:
                        if st.button("返回列表", use_container_width=True, type="tertiary"):
                            st.session_state['active_question_id'] = None
                            st.session_state['ai_result'] = None
                            st.rerun()
                    with col_prev:
                        if prev_q_id:
                            if st.button("上一题", use_container_width=True, type="tertiary"):
                                st.session_state['active_question_id'] = prev_q_id
                                st.session_state['ai_result'] = None
                                st.rerun()
                    with col_next:
                        if next_q_id:
                            if st.button("下一题", use_container_width=True, type="tertiary"):
                                st.session_state['active_question_id'] = next_q_id
                                st.session_state['ai_result'] = None
                                st.rerun()
                        
                    st.markdown("---")
                    
                    col_left, col_right = st.columns([1, 1.2])
                    with col_left:
                        with st.container(height=200, border=True):
                            st.info(active_q['question_text'])
                        
                        with st.container(height=520, border=True):
                            if 'ai_result' in st.session_state and st.session_state['ai_result'] is not None:
                                vlm_res = st.session_state['ai_result']
                                st.success("✅ 批改完成，成绩已计入全局学情统计！")
                                met_c1, met_c2 = st.columns(2)
                                met_c1.metric("🏅 智能总评分", f"{vlm_res['academic_correction']['score']} 分")
                                weak_points = vlm_res['academic_correction']['weak_knowledge_points']
                                met_c2.metric("🎯 核心薄弱点", f"{', '.join(weak_points) if weak_points else '无'}")
                                st.markdown(f"**过程得分明细**：`{vlm_res['academic_correction']['process_points']}`")
                                st.info(vlm_res['generated_response'])
                        
                    with col_right:
                        with st.container(height=736, border=True):
                            col_pen1, col_pen2, col_pen3, col_pen4 = st.columns([1.2, 1.5, 1, 1.2])
                            with col_pen1:
                                drawing_tool = st.selectbox("工具", ["✏️ 画笔", "🧽 橡皮擦"], key="global_tool")
                            with col_pen2:
                                stroke_width = st.slider("粗细", 1, 30, 3 if drawing_tool == "✏️ 画笔" else 20, key="global_width")
                            with col_pen3:
                                if drawing_tool == "✏️ 画笔":
                                    stroke_color = st.color_picker("颜色", "#000000", key="global_color")
                                else:
                                    stroke_color = "#ffffff"
                                    st.write("")
                                    st.markdown("<div style='margin-top: 10px; font-size: 14px; color: #888;'>已启用橡皮擦</div>", unsafe_allow_html=True)
                            with col_pen4:
                                st.write("")
                                st.write("")
                                if st.button("🗑️ 清空草稿", use_container_width=True, key="global_clear_btn"):
                                    st.session_state['global_clear_count'] = st.session_state.get('global_clear_count', 0) + 1
                                    st.rerun()
                                    
                            current_clear_count = st.session_state.get('global_clear_count', 0)
                                
                            canvas_result = st_canvas(
                                fill_color="rgba(255, 165, 0, 0.3)",
                                stroke_width=stroke_width,
                                stroke_color=stroke_color,
                                background_color="#ffffff",
                                height=580,
                                width=600,
                                update_streamlit=False,
                                drawing_mode="freedraw",
                                key=f"global_canvas_{current_clear_count}",
                            )
                        
                        if st.button("提交草稿给 AI 批改", type="primary", use_container_width=True):
                            if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
                                img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                                img = img.convert('RGB')
                                img_byte_arr = io.BytesIO()
                                img.save(img_byte_arr, format='JPEG')
                                ans_bytes = img_byte_arr.getvalue()
                                
                                with st.spinner("AI 正在解析您的草稿字迹并进行严谨批改..."):
                                    vlm_res = get_real_vlm_analysis(ans_bytes, None, global_subject, question_context=active_q['question_text'])
                                    
                                    update_profile(
                                        student_id_val, 
                                        vlm_res['visual_psych_scan']['emotional_inference'], 
                                        vlm_res['academic_correction']['correctness'],
                                        global_subject,
                                        vlm_res['academic_correction']['score'],
                                        vlm_res['academic_correction']['weak_knowledge_points']
                                    )
                                    
                                    st.session_state['ai_result'] = vlm_res
                                    st.rerun()
                            else:
                                st.warning("画板似乎是空的，请先书写解答。")
            
            elif st.session_state['active_folder'] is not None:
                # 文件夹详情页 (干净的题目列表)
                active_pdf = st.session_state['active_folder']
                
                if st.button("返回题库书架", type="tertiary"):
                    st.session_state['active_folder'] = None
                    st.rerun()
                    
                conn = sqlite3.connect('xiewo_student_archive.db')
                df_folder = pd.read_sql_query(f"SELECT * FROM question_bank WHERE student_id='{student_id_val}' AND subject='{global_subject}' AND source_pdf='{active_pdf}' ORDER BY id ASC", conn)
                conn.close()
                
                for idx, row in df_folder.iterrows():
                    q_preview = row['question_text'].replace('\n', ' ')
                    if len(q_preview) > 80:
                        q_preview = q_preview[:80] + "..."
                    if st.button(f"题目 {row['question_id']} (P{row['page_num']}): {q_preview}", key=f"go_{row['question_id']}", use_container_width=True, type="tertiary"):
                        st.session_state['active_question_id'] = row['question_id']
                        st.session_state['ai_result'] = None
                        st.rerun()
                    
            else:
                # 主书架模式
                uploaded_pdf = st.file_uploader("上传 PDF 提取新题目", type=['pdf'])
                if uploaded_pdf:
                    try:
                        pdf_bytes = uploaded_pdf.read()
                        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        total_pages = len(doc)
                        st.info(f"成功加载 PDF，共 {total_pages} 页。")
                        
                        # 页码范围选择器
                        if total_pages == 1:
                            page_range = (1, 1)
                            st.caption("本 PDF 仅 1 页。")
                        else:
                            page_range = st.slider(
                                "选择提取页码范围",
                                min_value=1, max_value=total_pages,
                                value=(1, total_pages),
                                key="page_range_slider"
                            )
                        
                        start_page, end_page = page_range
                        num_pages_to_extract = end_page - start_page + 1
                        
                        # 存入哪本图书？
                        conn_check = sqlite3.connect('xiewo_student_archive.db')
                        existing_books = pd.read_sql_query(
                            f"SELECT DISTINCT source_pdf FROM question_bank WHERE student_id='{student_id_val}' AND subject='{global_subject}'", 
                            conn_check
                        )['source_pdf'].tolist()
                        conn_check.close()
                        
                        if existing_books:
                            save_mode = st.radio("存入方式", ["📕 新建图书", "📂 存入已有图书"], horizontal=True, key="save_mode_radio")
                        else:
                            save_mode = "📕 新建图书"
                        
                        if save_mode == "📕 新建图书":
                            target_book_name = st.text_input("图书名称", value=uploaded_pdf.name.replace('.pdf', ''), key="new_book_name")
                        else:
                            target_book_name = st.selectbox("选择已有图书", existing_books, key="existing_book_select")
                        
                        if st.button(f"提取第 {start_page}-{end_page} 页（共 {num_pages_to_extract} 页，消耗 {num_pages_to_extract} 次额度）", type="primary", use_container_width=True):
                            progress_text = "AI 正在逐页提取题目，请耐心等待..."
                            my_bar = st.progress(0, text=progress_text)
                            
                            conn = sqlite3.connect('xiewo_student_archive.db')
                            c = conn.cursor()
                            
                            extracted_count = 0
                            for idx, i in enumerate(range(start_page - 1, end_page)):
                                my_bar.progress(idx / num_pages_to_extract, text=f"正在提取第 {i+1}/{total_pages} 页...")
                                page = doc.load_page(i)
                                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                                img_data = pix.tobytes("jpeg")
                                img_b64 = base64.b64encode(img_data).decode("utf-8")
                                
                                questions = extract_questions_from_page(img_b64, global_subject)
                                if questions:
                                    extracted_count += len(questions)
                                    for q in questions:
                                        qid = str(uuid.uuid4())[:8]
                                        q_text = str(q.get("content", q))
                                        c.execute("INSERT INTO question_bank (student_id, subject, source_pdf, page_num, question_text, question_id) VALUES (?, ?, ?, ?, ?, ?)",
                                                  (student_id_val, global_subject, target_book_name, i+1, q_text, qid))
                                    conn.commit()
                            conn.close()
                            my_bar.progress(1.0, text=f"🎉 提取完成！共提取 {extracted_count} 道新题目。")
                            time.sleep(1.5)
                            st.rerun()
                    except Exception as e:
                        st.error(f"解析 PDF 失败: {str(e)}")
                        
                st.divider()
                st.markdown("### 我的题库")
                conn = sqlite3.connect('xiewo_student_archive.db')
                df_qb = pd.read_sql_query(f"SELECT DISTINCT source_pdf FROM question_bank WHERE student_id='{student_id_val}' AND subject='{global_subject}'", conn)
                
                if df_qb.empty:
                    st.info("当前题库为空，请在上方上传 PDF 提取题目。")
                    conn.close()
                else:
                    for idx, row in df_qb.iterrows():
                        pdf_name = row['source_pdf']
                        c = conn.cursor()
                        c.execute("SELECT COUNT(*) FROM question_bank WHERE student_id=? AND subject=? AND source_pdf=?", (student_id_val, global_subject, pdf_name))
                        q_count = c.fetchone()[0]
                        
                        col_folder, col_edit = st.columns([6, 1])
                        with col_folder:
                            if st.button(f"【{pdf_name}】 (共 {q_count} 题)", key=f"open_{pdf_name}", use_container_width=True, type="tertiary"):
                                st.session_state['active_folder'] = pdf_name
                                st.rerun()
                        with col_edit:
                            if st.button("重命名", key=f"edit_{pdf_name}", type="tertiary"):
                                st.session_state["renaming_folder"] = pdf_name
                                st.rerun()
                                
                        if st.session_state.get("renaming_folder") == pdf_name:
                            new_name = st.text_input("请输入新的文件夹名称 (回车保存):", value=pdf_name, key=f"new_name_{pdf_name}")
                            if new_name and new_name != pdf_name:
                                c.execute("UPDATE question_bank SET source_pdf=? WHERE student_id=? AND subject=? AND source_pdf=?", (new_name, student_id_val, global_subject, pdf_name))
                                conn.commit()
                                st.session_state["renaming_folder"] = None
                                st.rerun()
                            if st.button("取消", key=f"cancel_{pdf_name}", type="tertiary"):
                                st.session_state["renaming_folder"] = None
                                st.rerun()
                    conn.close()

    elif st.session_state['role'] == 'teacher':
        # --- 教师端视图 ---
        if global_subject == "全部":
            st.header("📊 班级全局学情监控中枢 (所有科目)")
        else:
            st.header(f"📊 班级全局学情监控中枢 ({global_subject})")
            
        st.markdown("通过每一次日常作业批改，精准捕获全体学生的知识点盲区，**不落下一人。**")
        
        conn = sqlite3.connect('xiewo_student_archive.db')
        if global_subject == "全部":
            df = pd.read_sql_query("SELECT * FROM evaluation_history", conn)
        else:
            df = pd.read_sql_query(f"SELECT * FROM evaluation_history WHERE subject='{global_subject}'", conn)
        conn.close()
        
        if not df.empty:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("班级高频薄弱知识点雷达区")
                all_weak_points = []
                for wp_str in df['weak_points'].dropna():
                    if wp_str:
                        all_weak_points.extend(wp_str.split(','))
                
                if all_weak_points:
                    wp_counts = pd.Series(all_weak_points).value_counts().reset_index()
                    wp_counts.columns = ['知识点', '错误频次']
                    st.bar_chart(data=wp_counts.set_index('知识点'), y_label="全班犯错人次", color="#FF4B4B")
                else:
                    st.write("暂无薄弱点数据，同学们表现很棒！")
                    
            with col_chart2:
                st.subheader("学生学科平均分对比")
                latest_scores = df.groupby(['student_id', 'subject'])['score'].mean().unstack().fillna(0)
                if not latest_scores.empty:
                    st.bar_chart(latest_scores)
                else:
                    st.write("暂无分数数据。")
                    
            st.subheader("📋 全班批改流转记录表")
            st.dataframe(df[['student_id', 'subject', 'score', 'weak_points', 'timestamp']], use_container_width=True)
            
            st.divider()
            if st.button("🗑️ ⚠️ 清空全班所有数据"):
                conn = sqlite3.connect('xiewo_student_archive.db')
                conn.execute("DELETE FROM evaluation_history")
                conn.execute("DELETE FROM student_profiles")
                conn.commit()
                conn.close()
                st.success("全班数据已清空！")
                time.sleep(1)
                st.rerun()
        else:
            st.info("💡 暂无任何班级作业数据产生。")