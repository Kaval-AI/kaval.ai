import { v4 as uuidv4 } from 'uuid';

export interface ChatItem {
  uuid: string;
  createdAt: Date;
  updatedAt: Date;
  title: string
}

export class ChatItemFactory {
  static create(title: string): ChatItem {
    return {
      uuid: uuidv4(),
      createdAt: new Date(),
      updatedAt: new Date(),
      title: title
    };
  }
}
