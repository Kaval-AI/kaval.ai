import { Component, EventEmitter, Output } from '@angular/core';
import { MatInputModule } from '@angular/material/input';
import { MatIconModule } from '@angular/material/icon';
import { FormsModule } from '@angular/forms';
import { CdkTextareaAutosize, TextFieldModule } from '@angular/cdk/text-field'; // Import TextFieldModule for cdkTextareaAutosize

@Component({
  selector: 'app-chat-user-input',
  standalone: true,
  imports: [
    MatInputModule,
    MatIconModule,
    FormsModule,
    TextFieldModule,
    CdkTextareaAutosize
  ],
  templateUrl: './chat-user-input.html',
  styleUrl: './chat-user-input.css'
})
export class ChatUserInput {
  @Output() textSubmitted = new EventEmitter<string>();

  messageContent: string = '';

  handleKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter') {
      if (!event.shiftKey) {
        event.preventDefault();
        this.submitUserInput();
      }
    }
  }

  submitUserInput(): void {
    if (this.messageContent.trim().length > 0) {
      this.textSubmitted.emit(this.messageContent.trim());
      this.messageContent = '';
    }
  }
}