export interface UserDetails {
  id: string
  email: string
  name: string
  is_admin: boolean
  picture: string
  active_project_id?: string
}
