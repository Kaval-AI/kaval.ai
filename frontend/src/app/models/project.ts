export interface Project {
  id: string;
  name: string;
  description?: string;
  created_at?: string;
  updated_at?: string;
  role?: 'owner' | 'viewer';
}
