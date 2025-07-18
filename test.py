from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()


supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
data = {
    'user_id': '42b49e49-6f6b-46ec-8b3e-bf4cfbd6585e',
    'type': 'request',
    'message': 'Test request',
    'status': 'pending',
    'priority': 'low',
    'created_at': '2024-07-19T12:00:00Z'
  }
print(supabase.table('requests').insert(data).execute())