#!/usr/bin/env python3
import asyncio
from graphrag_query import EmergencyMedicineGraphRAG

async def test_search():
    print('Initializing GraphRAG...')
    graph_rag = EmergencyMedicineGraphRAG()
    
    print('\nTesting improved local search...')
    result = await graph_rag.local_search('REBOA trauma surgical patients')
    print('Result:')
    print(result)

if __name__ == "__main__":
    asyncio.run(test_search())
