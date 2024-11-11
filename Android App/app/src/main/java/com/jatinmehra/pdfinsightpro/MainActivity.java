package com.jatinmehra.pdfinsightpro;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    private WebView myweb;
    private ValueCallback<Uri[]> filePathCallback;

    // File chooser launcher
    private final ActivityResultLauncher<Intent> fileChooserLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(), result -> {
                if (filePathCallback != null) {
                    Uri[] resultUris = null;
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        resultUris = new Uri[]{result.getData().getData()};
                    }
                    filePathCallback.onReceiveValue(resultUris);
                    filePathCallback = null;
                }
            }
    );

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize WebView
        myweb = findViewById(R.id.myweb);
        myweb.getSettings().setJavaScriptEnabled(true);

        // Set WebViewClient to handle all URL navigation
        myweb.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, String url) {
                // Handle mailto links
                if (url.startsWith("mailto:")) {
                    Intent intent = new Intent(Intent.ACTION_SENDTO);
                    intent.setData(Uri.parse(url)); // Open the default email app
                    startActivity(intent);
                    return true;
                }
                return super.shouldOverrideUrlLoading(view, url);
            }
        });

        // Set WebChromeClient to handle file chooser requests
        myweb.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onShowFileChooser(WebView webView, ValueCallback<Uri[]> filePathCallback, FileChooserParams fileChooserParams) {
                // Store the callback and open the file picker
                MainActivity.this.filePathCallback = filePathCallback;
                Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                intent.setType("application/pdf"); // Ensure PDF files are picked
                fileChooserLauncher.launch(intent);
                return true;
            }
        });

        // Load your Streamlit app URL
        myweb.loadUrl("https://jatinmehra-pdf-insight-pro.hf.space");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Clean up the filePathCallback to avoid memory leaks
        if (filePathCallback != null) {
            filePathCallback.onReceiveValue(null);
            filePathCallback = null;
        }
    }
}
