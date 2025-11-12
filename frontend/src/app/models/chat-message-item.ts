import { v4 as uuidv4 } from 'uuid';


export interface ChatMessageItem {
  id: string;
  chatId: string;
  agentRunId: string;
  role: string;
  content: string
  createdAt: Date;
  updatedAt: Date;
}

export class ChatMessageItemFactory {
  static create(chatId: string, role: string, content: string): ChatMessageItem {
    return {
      id: uuidv4(),
      chatId: chatId,
      role: role,
      agentRunId: "",
      content: content,
      createdAt: new Date(),
      updatedAt: new Date()
    };
  }
}
