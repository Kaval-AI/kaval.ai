export interface Project {
  id: string;
  name: string;
  description?: string;
  db_host?: string;
  db_port?: number;
  db_user?: string;
  db_password?: string;
  db_name?: string;
  db_schema?: string;
  created_at?: string;
  updated_at?: string;
  role?: 'owner' | 'viewer';
}
