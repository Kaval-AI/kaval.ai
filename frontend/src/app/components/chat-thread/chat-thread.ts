import { Component, Input, Output, ViewChild, ElementRef, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChatMessage } from '../chat-message/chat-message';
import { AgentRun } from '../agent-run/agent-run';
import { ChatUserInput } from '../chat-user-input/chat-user-input';
import { ChatMessageItem, ChatMessageItemFactory } from '../../models/chat-message-item';
import { ChatsService } from '../../services/chats-service';
import { NgZone } from '@angular/core';

@Component({
  selector: 'app-chat-thread',
  imports: [AgentRun, ChatUserInput, CommonModule],
  templateUrl: './chat-thread.html',
  styleUrl: './chat-thread.css'
})
export class ChatThread {
  @ViewChild('messagesContainer') private messagesContainerRef!: ElementRef;

  // List of chat messages and AI responses.
  messages: ChatMessageItem[] = [];

  agentRuns: any[] = []

  // Current chat ID, null if this is a new chat.
  private _currentChatId: string | null = null;

  @Input()
  set currentChatId(value: string | null) {
    if (value === null) {
      console.log("Clearing messages");
      this.messages = [];
    } else {
      this.chatsService.listMessages(value).then(msgs => {
        this.messages = [...msgs];
        console.log(`"Loaded ${this.messages.length} for chat`);
        this.scrollToBottom();
      });
      this.chatsService.listAgentRuns(value).then(runs => {
        this.agentRuns = [...runs];
        console.log(`"Loaded ${this.agentRuns.length} agent runs for chat.`);
        console.log(this.agentRuns)
      });
    }
    this._currentChatId = value;
  }

  get currentChatId(): string | null {
    return this._currentChatId;
  }

  // Should emit user inputted text if a new chat is initiated.
  @Output() newChatInitiated = new EventEmitter<{ chatUuid: string }>();

  constructor(private chatsService: ChatsService, private ngZone: NgZone) { }

  isEmptyChat(): boolean {
    if (this.currentChatId === null) {
      if (this.messages.length !== 0) {
        throw new Error("Invariant violation: messages should be empty for new chat");
      }
    }
    return this.messages.length === 0;
  }

  async onUserInputSubmitted(messageText: string): Promise<void> {
    //let userMessage = ChatMessageItemFactory.create("user", messageText);
    // If this is a new chat, ask the ChatService to create a new chat.
    if (this.isEmptyChat()) {
      let title: string = messageText.substring(0, 30);
      let chat = await this.chatsService.createChat(title);
      this._currentChatId = chat.id;
      this.newChatInitiated.emit({ chatUuid: chat.id });
    }

    // Creates user message.
    let userMessage = ChatMessageItemFactory.create(this.currentChatId!, "user", messageText);
    this.messages.push(userMessage);
    this.scrollToBottom();
    userMessage = await this.chatsService.addMessage(userMessage);

    // Creates assistant message.
    let assistantMessage = ChatMessageItemFactory.create(this.currentChatId!, "assistant", "");
    this.messages.push(assistantMessage);
    this.scrollToBottom();

    // Stream message generation.
    let responseObsservable = this.chatsService.generateResponse(this.currentChatId!)
    responseObsservable.subscribe({
      next: (response: string) => {
        this.ngZone.run(() => {
          assistantMessage.content = response;
          this.scrollToBottom();
        });
      },
      complete: () => {
        this.chatsService.addMessage(assistantMessage);
        console.log(`Response received, total of ${assistantMessage.content.length} characters`);
      }
    });
  }

  private scrollToBottom(): void {
    if (this.messagesContainerRef) {
      const container = this.messagesContainerRef.nativeElement;

      requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
      });
    }
  }
}
