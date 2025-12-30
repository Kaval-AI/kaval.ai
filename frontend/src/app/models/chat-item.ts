import { v4 as uuidv4 } from 'uuid';

export interface ChatItems {
  chats: ChatItem[];
}

export interface ChatItem {
  id: string;
  title: string;
  context: any;
  createdAt: Date;
  updatedAt: Date;
}

export class ChatItemFactory {
  static create(title: string): ChatItem {
    return {
      id: uuidv4(),
      title: title,
      context: null,
      createdAt: new Date(),
      updatedAt: new Date(),
    };
  }
}
