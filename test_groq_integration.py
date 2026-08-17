import asyncio
from api.index import thinker_analyze, manager

async def test():
    print("Testing thinker...")
    res = await thinker_analyze("hello")
    print(f"Thinker result: {res}")
    
    print("Testing handle_ai_chat...")
    class MockWS:
        async def send_json(self, data):
            print(f"WS SEND: {data}")
            
    manager.active_connections["test_id"] = MockWS()
    manager.user_data["test_id"] = {"history": [], "sub_county": "Test", "county": "Test"}
    
    await manager.handle_ai_chat("test_id", "hello", is_nudge=False, depth=0.0)
    print("Done testing.")

asyncio.run(test())
