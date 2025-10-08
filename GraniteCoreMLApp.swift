// GraniteCoreMLApp.swift
// Native macOS interface for IBM Granite 4.0 model with CoreML acceleration

import SwiftUI
import CoreML

// Main app structure
@main
struct GraniteCoreMLApp: App {
    @StateObject private var modelManager = GraniteModelManager()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(modelManager)
                .frame(minWidth: 800, minHeight: 600)
        }
        .windowStyle(HiddenTitleBarWindowStyle())
    }
}

class GraniteModelManager: ObservableObject {
    @Published var isModelLoaded = false
    @Published var isGenerating = false
    @Published var currentResponse = ""
    @Published var conversationHistory: [ChatMessage] = []
    @Published var errorMessage: String?
    
    private var model: MLModel?
    private var modelURL: URL?
    
    init() {
        loadModel()
    }
    
    func loadModel() {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                // Look for the CoreML model in the coreml_models directory
                let fileManager = FileManager.default
                let currentDirectory = fileManager.currentDirectoryPath
                let modelPath = "\(currentDirectory)/model/coreml_models/granite-4.0-h-micro.mlpackage"
                let modelURL = URL(fileURLWithPath: modelPath)
                
                if fileManager.fileExists(atPath: modelPath) {
                    self.model = try MLModel(contentsOf: modelURL)
                    self.modelURL = modelURL
                    
                    DispatchQueue.main.async {
                        self.isModelLoaded = true
                        self.errorMessage = nil
                    }
                } else {
                    DispatchQueue.main.async {
                        self.errorMessage = "CoreML model not found at \(modelPath). Please run the conversion script first."
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to load model: \(error.localizedDescription)"
                }
            }
        }
    }
    
    func sendMessage(_ text: String) {
        guard !text.isEmpty else { return }
        
        let userMessage = ChatMessage(content: text, isUser: true, timestamp: Date())
        conversationHistory.append(userMessage)
        
        isGenerating = true
        currentResponse = ""
        
        // For now, simulate a response since the actual CoreML integration needs more work
        DispatchQueue.global(qos: .userInitiated).async {
            self.simulateModelResponse(to: text)
        }
    }
    
    private func simulateModelResponse(to prompt: String) {
        // This is a placeholder - in the real implementation, this would call the CoreML model
        let responses = [
            "I'm a simulated response from the IBM Granite model. The actual CoreML integration would provide more sophisticated responses based on the trained model.",
            "This is a demonstration of the native macOS interface. Once the CoreML model is properly loaded and configured, this would generate responses using Apple's optimized inference engine.",
            "The interface is ready for the IBM Granite 4.0 model with CoreML acceleration. The model conversion process needs to complete first for full functionality."
        ]
        
        let randomResponse = responses.randomElement() ?? "Response generation in progress..."
        
        DispatchQueue.main.async {
            self.currentResponse = randomResponse
            let assistantMessage = ChatMessage(content: randomResponse, isUser: false, timestamp: Date())
            self.conversationHistory.append(assistantMessage)
            self.isGenerating = false
        }
    }
    
    func clearConversation() {
        conversationHistory.removeAll()
        currentResponse = ""
    }
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let content: String
    let isUser: Bool
    let timestamp: Date
}

struct ContentView: View {
    @EnvironmentObject var modelManager: GraniteModelManager
    @State private var inputText = ""
    @FocusState private var isInputFocused: Bool
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            headerView
            
            // Chat messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 16) {
                        ForEach(modelManager.conversationHistory) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }
                        
                        if modelManager.isGenerating {
                            TypingIndicator()
                        }
                    }
                    .padding()
                }
                .onChange(of: modelManager.conversationHistory.count) { _ in
                    if let lastMessage = modelManager.conversationHistory.last {
                        withAnimation {
                            proxy.scrollTo(lastMessage.id, anchor: .bottom)
                        }
                    }
                }
            }
            
            // Input area
            inputArea
        }
        .background(Color(NSColor.windowBackgroundColor))
        .alert(item: $modelManager.errorMessage) { error in
            Alert(
                title: Text("Model Error"),
                message: Text(error),
                dismissButton: .default(Text("OK"))
            )
        }
    }
    
    private var headerView: some View {
        HStack {
            Image(systemName: "cpu")
                .font(.title)
                .foregroundColor(.accentColor)
            
            VStack(alignment: .leading) {
                Text("IBM Granite 4.0")
                    .font(.title2)
                    .fontWeight(.bold)
                Text("CoreML Accelerated")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            HStack {
                Circle()
                    .fill(modelManager.isModelLoaded ? Color.green : Color.red)
                    .frame(width: 12, height: 12)
                
                Text(modelManager.isModelLoaded ? "Model Ready" : "Loading Model...")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Button(action: modelManager.clearConversation) {
                Image(systemName: "trash")
                    .foregroundColor(.secondary)
            }
            .buttonStyle(PlainButtonStyle())
            .help("Clear conversation")
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
    }
    
    private var inputArea: some View {
        HStack(spacing: 12) {
            TextField("Type your message...", text: $inputText)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .focused($isInputFocused)
                .disabled(modelManager.isGenerating)
                .onSubmit {
                    sendMessage()
                }
            
            Button(action: sendMessage) {
                HStack {
                    Image(systemName: "paperplane.fill")
                    Text("Send")
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
            }
            .disabled(inputText.isEmpty || modelManager.isGenerating)
            .buttonStyle(BorderedProminentButtonStyle())
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
    }
    
    private func sendMessage() {
        guard !inputText.isEmpty else { return }
        modelManager.sendMessage(inputText)
        inputText = ""
        isInputFocused = false
    }
}

struct MessageBubble: View {
    let message: ChatMessage
    
    var body: some View {
        HStack {
            if message.isUser {
                Spacer()
            }
            
            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 4) {
                Text(message.content)
                    .padding(12)
                    .background(message.isUser ? Color.accentColor : Color(NSColor.controlBackgroundColor))
                    .foregroundColor(message.isUser ? .white : .primary)
                    .cornerRadius(16)
                    .frame(maxWidth: 400, alignment: message.isUser ? .trailing : .leading)
                
                Text(message.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            if !message.isUser {
                Spacer()
            }
        }
    }
}

struct TypingIndicator: View {
    @State private var dotCount = 0
    
    var body: some View {
        HStack {
            Spacer()
            
            HStack(spacing: 4) {
                Text("IBM Granite is thinking")
                    .foregroundColor(.secondary)
                
                ForEach(0..<3) { index in
                    Circle()
                        .fill(Color.secondary)
                        .frame(width: 6, height: 6)
                        .opacity((dotCount / 3) == index ? 1 : 0.3)
                }
            }
            .padding(12)
            .background(Color(NSColor.controlBackgroundColor))
            .cornerRadius(16)
            
            Spacer()
        }
        .onAppear {
            Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
                dotCount = (dotCount + 1) % 12
            }
        }
    }
}

// Extension to make String conform to Identifiable for alert
extension String: Identifiable {
    public var id: String { self }
}