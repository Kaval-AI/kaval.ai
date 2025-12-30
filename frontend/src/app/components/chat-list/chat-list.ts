import {
  Component,
  Output,
  EventEmitter,
  OnChanges,
  SimpleChanges,
  OnInit,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatMenuModule } from '@angular/material/menu';
import { MatDialog } from '@angular/material/dialog';
import { RenameChatDialog } from '../rename-chat-dialog/rename-chat-dialog';
import { DeleteChatDialog } from '../delete-chat-dialog/delete-chat-dialog';
import { ChatItem } from '../../models/chat-item';
import { ChatsService } from '../../services/chats-service';

@Component({
  selector: 'app-chat-list',
  standalone: true,
  imports: [CommonModule, MatButtonModule, MatIconModule, MatMenuModule],
  templateUrl: './chat-list.html',
  styleUrl: './chat-list.css',
})
export class ChatList implements OnInit {
  chats: ChatItem[] = [];

  hoveredChatId: string | null = null;
  selectedChatId: string | null = null;
  isCondensed: boolean = false;

  // User has requested a new empty chat.
  @Output() newChatRequested = new EventEmitter<void>();
  // User has clicked on a chat.
  @Output() chatSelected = new EventEmitter<string>();

  constructor(
    public dialog: MatDialog,
    public chatsService: ChatsService
  ) {}

  ngOnInit(): void {
    this.reloadChats();
  }

  onNewChatClicked(): void {
    this.newChatRequested.emit();
    this.selectedChatId = null;
  }

  reloadChats(): void {
    this.chatsService.listChats().then((chats) => (this.chats = chats));
  }

  getOrderedChats(): ChatItem[] {
    return [...this.chats].sort(
      (a, b) =>
        new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    );
  }

  onChatClick(chatId: string): void {
    console.log('Chat selected:', chatId);
    this.selectedChatId = chatId;
    this.chatSelected.emit(chatId);
  }

  onMouseEnter(chatId: string): void {
    this.hoveredChatId = chatId;
  }

  onMouseLeave(): void {
    this.hoveredChatId = null;
  }

  onRenameChat(chatId: string): void {
    const chat = this.chats.find((c) => c.id === chatId);
    if (!chat) return;

    const dialogRef = this.dialog.open(RenameChatDialog, {
      width: '600px',
      data: { currentTitle: chat.title },
    });

    dialogRef.afterClosed().subscribe((result) => {
      if (result) {
        chat.title = result;
        chat.updatedAt = new Date();
      }
    });
  }

  onDeleteChat(chatId: string): void {
    const chat = this.chats.find((c) => c.id === chatId);
    if (!chat) return;

    const dialogRef = this.dialog.open(DeleteChatDialog, {
      width: '600px',
      data: { title: chat.title },
    });

    dialogRef.afterClosed().subscribe((result) => {
      if (result === true) {
        this.chats = this.chats.filter((c) => c.id !== chatId);
      }
    });
  }

  // Method to emit the request to toggle the sidenav
  toggleCondensed(): void {
    this.isCondensed = !this.isCondensed;
  }
}
