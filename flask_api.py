from flask import Flask, request, jsonify
import requests
import json
import os
from datetime import datetime

app = Flask(__name__)

# 内存中的简单数据库
tasks = [
    {"id": 1, "title": "学习Python", "completed": False, "created_at": "2023-01-01T10:00:00Z"},
    {"id": 2, "title": "构建API", "completed": True, "created_at": "2023-01-02T14:30:00Z"},
    {"id": 3, "title": "数据可视化", "completed": False, "created_at": "2023-01-03T09:15:00Z"}
]

# 获取所有任务
@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    status = request.args.get('status')
    
    if status:
        if status.lower() == 'completed':
            filtered_tasks = [task for task in tasks if task['completed']]
        elif status.lower() == 'pending':
            filtered_tasks = [task for task in tasks if not task['completed']]
        else:
            return jsonify({"error": "状态参数无效，请使用 'completed' 或 'pending'"}), 400
        
        return jsonify(filtered_tasks)
    
    return jsonify(tasks)

# 获取单个任务
@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = next((task for task in tasks if task['id'] == task_id), None)
    
    if task:
        return jsonify(task)
    
    return jsonify({"error": f"任务ID {task_id} 不存在"}), 404

# 创建任务
@app.route('/api/tasks', methods=['POST'])
def create_task():
    data = request.get_json()
    
    if not data or 'title' not in data:
        return jsonify({"error": "请提供任务标题"}), 400
    
    # 生成新ID (简单实现)
    new_id = max(task['id'] for task in tasks) + 1 if tasks else 1
    
    new_task = {
        "id": new_id,
        "title": data['title'],
        "completed": data.get('completed', False),
        "created_at": datetime.now().isoformat() + 'Z'
    }
    
    tasks.append(new_task)
    return jsonify(new_task), 201

# 更新任务
@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    task = next((task for task in tasks if task['id'] == task_id), None)
    
    if not task:
        return jsonify({"error": f"任务ID {task_id} 不存在"}), 404
    
    data = request.get_json()
    
    if 'title' in data:
        task['title'] = data['title']
    
    if 'completed' in data:
        task['completed'] = data['completed']
    
    return jsonify(task)

# 删除任务
@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    task = next((task for task in tasks if task['id'] == task_id), None)
    
    if not task:
        return jsonify({"error": f"任务ID {task_id} 不存在"}), 404
    
    tasks.remove(task)
    return jsonify({"message": f"任务ID {task_id} 已删除"}), 200

# 从外部API获取天气
@app.route('/api/weather/<city>', methods=['GET'])
def get_weather(city):
    try:
        # 注意：需要一个有效的API密钥 - 这只是示例
        api_key = os.environ.get('WEATHER_API_KEY', 'your_api_key_here')
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=zh_cn"
        
        response = requests.get(url)
        data = response.json()
        
        if response.status_code != 200:
            return jsonify({"error": data.get('message', '获取天气数据失败')}), response.status_code
        
        weather_data = {
            "城市": city,
            "温度": data['main']['temp'],
            "描述": data['weather'][0]['description'],
            "湿度": data['main']['humidity']
        }
        
        return jsonify(weather_data)
    
    except Exception as e:
        return jsonify({"error": f"获取天气数据时出错: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
