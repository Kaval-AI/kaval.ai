import { v4 as uuidv4 } from 'uuid';

export interface ChatMessageItem {
  uuid: string;
  chatUuid: string;
  role: string;
  content: string
  createdAt: Date;
  updatedAt: Date;
}

export class ChatMessageItemFactory {
  static create(chatUuid: string, role: string, content: string): ChatMessageItem {
    return {
      uuid: uuidv4(),
      chatUuid: chatUuid,
      role,
      content,
      createdAt: new Date(),
      updatedAt: new Date()
    };
  }
}
