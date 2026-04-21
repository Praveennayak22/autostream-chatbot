import sys, os
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, '.')
os.chdir(r'c:\Users\prave\Downloads\ServiceHive')

from agent.graph import graph
from langchain_core.messages import HumanMessage, AIMessage

def run_turn(state, user_text):
    state['messages'] = state['messages'] + [HumanMessage(content=user_text)]
    new = graph.invoke(state)
    ai = [m for m in new['messages'] if isinstance(m, AIMessage)]
    intent = new['intent']
    lead = new.get('lead_info')
    collecting = new.get('collecting_lead')
    captured = new.get('lead_captured')
    reply = ai[-1].content[:200] if ai else '(no reply)'
    print(f'\n[USER]    {user_text}')
    print(f'[INTENT]  {intent}')
    print(f'[AGENT]   {reply}')
    print(f'[LEAD]    {lead}  collecting={collecting}  captured={captured}')
    return new

state = {'messages': [], 'intent': '', 'lead_info': {}, 'lead_captured': False, 'collecting_lead': False}

print('=== Turn 1: Greeting ===')
state = run_turn(state, 'Hi!')

print('\n=== Turn 2: Product inquiry ===')
state = run_turn(state, 'What are your pricing plans?')

print('\n=== Turn 3: High intent ===')
state = run_turn(state, 'I want to try the Pro plan for my YouTube channel.')

print('\n=== Turn 4: Name ===')
state = run_turn(state, 'My name is Alex')

print('\n=== Turn 5: Email ===')
state = run_turn(state, 'alex@example.com')

print('\n=== Turn 6: Platform (lead capture) ===')
state = run_turn(state, 'YouTube')
