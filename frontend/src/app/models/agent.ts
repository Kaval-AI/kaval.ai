export interface Agent {
  id: string;
  name: string;
  description?: string;
  input_schema?: any;
  output_schema?: any;
  workflow?: any;
  created_at?: string;
  updated_at?: string;
}
