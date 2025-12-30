import { Component, inject, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChatList } from '../chat-list/chat-list';
import { ChatThread } from '../chat-thread/chat-thread';
import { ChatsService } from '../../services/chats-service';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, ChatList, ChatThread],
  templateUrl: './chat.html',
  styleUrl: './chat.css',
})
export class Chat {
  @ViewChild('chatThread') private chatThread!: ChatThread;
  @ViewChild('chatList') private chatList!: ChatList;

  constructor(public chatsService: ChatsService) {}

  /**
   * Called when current chat should be cleared (e.g. user wants to start a new chat).
   */
  onNewChatRequested(): void {
    this.chatThread.currentChatId = null;
  }

  /**
   * Called after user has initiated a conversation in an empty chat thread.
   */
  onNewChatInitiated(value: { chatUuid: string }): void {
    this.chatList.reloadChats();
  }

  /**
   * Called when user selects a chat from the list.
   * @param uuid UUID of the chat selected by user.
   */
  onChatSelected(uuid: string): void {
    this.chatThread.currentChatId = uuid;
  }
}
